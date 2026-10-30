#!/usr/bin/env python3
"""Standalone CPU checks for full-sequence causal multi-head self-attention."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def split_heads(x, heads):
    batch, length, width = x.shape
    return x.reshape(batch, length, heads, width // heads).transpose(1, 2)


def merge_heads(x):
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, length, heads * head_dim)


def manual_attention(q, k, v, causal=True, softmax_dim=-1):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    allowed = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool).tril()
    if causal:
        scores = scores.masked_fill(~allowed, float('-inf'))
    weights = scores.softmax(dim=softmax_dim)
    return weights @ v, weights


def mha(x, wq, wk, wv, wo, heads, reference=False, causal=True):
    q, k, v = (split_heads(x @ w, heads) for w in (wq, wk, wv))
    if reference:
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
        weights = None
    else:
        z, weights = manual_attention(q, k, v, causal=causal)
    return merge_heads(z) @ wo, weights


def max_error(a, b):
    return float((a.detach() - b.detach()).abs().max())


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    batch, length, width, heads = (2, 5, 12, 3) if quick else (3, 7, 24, 4)
    dtype = torch.float64
    x = torch.randn(batch, length, width, dtype=dtype)
    matrices = [torch.randn(width, width, dtype=dtype) / math.sqrt(width) for _ in range(4)]
    manual_leaves = [value.clone().requires_grad_(True) for value in (x, *matrices)]
    reference_leaves = [value.clone().requires_grad_(True) for value in (x, *matrices)]
    out, weights = mha(*manual_leaves, heads)
    ref, _ = mha(*reference_leaves, heads, reference=True)
    upstream = torch.randn_like(out)
    manual_grads = torch.autograd.grad((out * upstream).sum(), manual_leaves)
    reference_grads = torch.autograd.grad((ref * upstream).sum(), reference_leaves)
    forward_error = max_error(out, ref)
    gradient_errors = {
        name: max_error(a, b)
        for name, a, b in zip(('input', 'wq', 'wk', 'wv', 'wo'), manual_grads, reference_grads)
    }
    upper = torch.ones(length, length, dtype=torch.bool).triu(diagonal=1)
    row_error = float((weights.detach().sum(dim=-1) - 1).abs().max())
    future_weight = float(weights.detach()[..., upper].abs().max())

    boundary = length // 2
    perturbed = x.clone()
    perturbed[:, boundary:] += 3 * torch.randn_like(perturbed[:, boundary:])
    original, _ = mha(x, *matrices, heads)
    changed, _ = mha(perturbed, *matrices, heads)
    prefix_error = max_error(original[:, :boundary], changed[:, :boundary])
    suffix_difference = max_error(original[:, boundary:], changed[:, boundary:])
    unmasked, _ = mha(x, *matrices, heads, causal=False)
    unmasked_changed, _ = mha(perturbed, *matrices, heads, causal=False)
    leaked_difference = max_error(unmasked[:, :boundary], unmasked_changed[:, :boundary])

    causal_input = x.clone().requires_grad_(True)
    causal_output, _ = mha(causal_input, *matrices, heads)
    dependency_grad = torch.autograd.grad(causal_output[:, :boundary].square().sum(), causal_input)[0]
    future_gradient = float(dependency_grad[:, boundary:].abs().max())

    q, k, v = (split_heads(x @ w, heads) for w in matrices[:3])
    wrong_axis, wrong_weights = manual_attention(q, k, v, softmax_dim=-2)
    wrong_axis_row_error = float((wrong_weights.sum(dim=-1) - 1).abs().max())
    correct_heads, _ = manual_attention(q, k, v)
    wrong_merge = correct_heads.contiguous().reshape(batch, length, width)
    wrong_merge_error = max_error(wrong_merge, merge_heads(correct_heads))
    wrong_axis_output_error = max_error(wrong_axis, correct_heads)
    # Float64 CPU comparison; no claim about FP16/BF16 tolerances or GPU kernels.
    tolerance = 1e-10
    checks = {
        'output_shape': list(out.shape) == [batch, length, width],
        'finite_output_and_gradients': bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(g).all()) for g in manual_grads),
        'sdpa_forward_matches': forward_error < tolerance,
        'sdpa_all_five_gradients_match': all(error < tolerance for error in gradient_errors.values()),
        'key_axis_probabilities_sum_to_one': row_error < tolerance,
        'future_attention_weights_are_zero': future_weight == 0.0,
        'first_query_reads_only_first_key': bool((weights.detach()[..., 0, 0] == 1).all()),
        'future_input_perturbation_preserves_prefix': prefix_error < tolerance,
        'perturbation_changes_visible_suffix': suffix_difference > 1e-6,
        'future_input_gradient_is_zero': future_gradient == 0.0,
        'negative_control_no_mask_leaks': leaked_difference > 1e-6,
        'negative_control_wrong_softmax_axis_detected': wrong_axis_row_error > 1e-6 and wrong_axis_output_error > 1e-6,
        'negative_control_missing_transpose_detected': wrong_merge_error > 1e-6,
    }
    report = {
        'chapter': '02-attention', 'seed': seed, 'checks': checks,
        'metrics': {
            'dtype': 'float64', 'device': 'cpu', 'batch': batch, 'length': length,
            'width': width, 'heads': heads, 'head_dim': width // heads,
            'sdpa_forward_max_abs_error': forward_error,
            'sdpa_gradient_max_abs_errors': gradient_errors,
            'attention_row_sum_max_abs_error': row_error,
            'future_attention_max_abs_weight': future_weight,
            'causal_prefix_max_abs_error': prefix_error,
            'visible_suffix_max_abs_change': suffix_difference,
            'future_input_gradient_max_abs': future_gradient,
            'unmasked_prefix_max_abs_change': leaked_difference,
            'wrong_softmax_row_sum_max_abs_error': wrong_axis_row_error,
            'wrong_softmax_output_max_abs_error': wrong_axis_output_error,
            'wrong_head_merge_max_abs_error': wrong_merge_error,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args.seed, args.quick)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    if not all(report['checks'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
