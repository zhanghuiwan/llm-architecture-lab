#!/usr/bin/env python3
"""Grouped-query attention: head mapping, gradients and theoretical KV budgets."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def split_heads(x, weight, heads, head_dim):
    batch, time, _ = x.shape
    return (x @ weight).reshape(batch, time, heads, head_dim).transpose(1, 2)


def validate(hidden, query_heads, kv_heads):
    if hidden <= 0 or query_heads <= 0 or kv_heads <= 0:
        raise ValueError('Dimensions and head counts must be positive')
    if hidden % query_heads or query_heads % kv_heads:
        raise ValueError('This experiment requires D % Hq == 0 and Hq % Hkv == 0')


def attention(x, wq, wk, wv, wo, query_heads, kv_heads, mode='grouped'):
    batch, time, hidden = x.shape
    validate(hidden, query_heads, kv_heads)
    head_dim = hidden // query_heads
    repeats = query_heads // kv_heads
    q = split_heads(x, wq, query_heads, head_dim)
    k = split_heads(x, wk, kv_heads, head_dim)
    v = split_heads(x, wv, kv_heads, head_dim)
    allowed = torch.ones(time, time, dtype=torch.bool, device=x.device).tril()
    if mode == 'grouped':
        # Hq is ordered as [kv group, query within group]. K/V stay compact.
        qg = q.reshape(batch, kv_heads, repeats, time, head_dim)
        scores = torch.einsum('bgrtd,bgsd->bgrts', qg, k) / head_dim**0.5
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        out = torch.einsum('bgrts,bgsd->bgrtd', probabilities, v)
        out = out.reshape(batch, query_heads, time, head_dim)
    elif mode == 'repeated':
        kr = k.repeat_interleave(repeats, dim=1)
        vr = v.repeat_interleave(repeats, dim=1)
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, dropout_p=0.0)
    elif mode == 'head_loop':
        outputs = []
        for head in range(query_heads):
            group = head // repeats
            scores = q[:, head] @ k[:, group].transpose(-2, -1) / head_dim**0.5
            p = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
            outputs.append(p @ v[:, group])
        out = torch.stack(outputs, dim=1)
    elif mode == 'wrong_repeat':
        # Same shape, different mapping: [0,1,0,1,...] instead of [0,0,...,1,1,...].
        kr = k.repeat(1, repeats, 1, 1)
        vr = v.repeat(1, repeats, 1, 1)
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, dropout_p=0.0)
    else:
        raise ValueError(mode)
    return out.transpose(1, 2).contiguous().reshape(batch, time, hidden) @ wo


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    batch, time, hidden, query_heads = (2, 5, 24, 6) if quick else (3, 7, 48, 8)
    head_dim = hidden // query_heads
    kv_cases = (1, 2, 6) if quick else (1, 2, 4, 8)
    checks, rows = {}, []

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    check('sequence_and_head_dimensions_differ', time != head_dim)
    for bad in ((hidden, query_heads, 0), (hidden, query_heads, query_heads + 1),
                (hidden + 1, query_heads, 1)):
        rejected = False
        try:
            validate(*bad)
        except ValueError:
            rejected = True
        check('reject_invalid_' + '_'.join(map(str, bad)), rejected)

    layers, element_bytes = 4, torch.tensor([], dtype=dtype).element_size()
    mha_cache = 2 * batch * layers * time * query_heads * head_dim * element_bytes
    for kv_heads in kv_cases:
        repeats = query_heads // kv_heads
        x = torch.randn(batch, time, hidden, dtype=dtype, requires_grad=True)
        weights = [
            (torch.randn(hidden, query_heads * head_dim, dtype=dtype) / hidden**0.5).requires_grad_(),
            (torch.randn(hidden, kv_heads * head_dim, dtype=dtype) / hidden**0.5).requires_grad_(),
            (torch.randn(hidden, kv_heads * head_dim, dtype=dtype) / hidden**0.5).requires_grad_(),
            (torch.randn(query_heads * head_dim, hidden, dtype=dtype) / hidden**0.5).requires_grad_(),
        ]
        arguments = [x, *weights]
        reference_arguments = [a.detach().clone().requires_grad_() for a in arguments]
        out = attention(*arguments, query_heads, kv_heads)
        reference = attention(*reference_arguments, query_heads, kv_heads, mode='repeated')
        head_loop = attention(*arguments, query_heads, kv_heads, mode='head_loop')
        prefix = f'kv{kv_heads}_'
        check(prefix + 'output_shape', list(out.shape) == [batch, time, hidden])
        forward_error = float((out.detach() - reference.detach()).abs().max())
        check(prefix + 'repeated_reference_forward', forward_error < 1e-10)
        check(prefix + 'explicit_head_mapping', torch.allclose(out, head_loop, atol=1e-10, rtol=1e-10))
        probe = torch.randn_like(out)
        gradients = torch.autograd.grad((out * probe).sum(), arguments)
        reference_gradients = torch.autograd.grad((reference * probe).sum(), reference_arguments)
        gradient_errors = {}
        for name, actual, expected in zip(('x', 'wq', 'wk', 'wv', 'wo'), gradients, reference_gradients):
            gradient_errors[name] = float((actual - expected).abs().max())
            check(prefix + 'gradient_' + name, torch.allclose(actual, expected, atol=1e-9, rtol=1e-9))
            check(prefix + 'finite_nonzero_gradient_' + name,
                  bool(torch.isfinite(actual).all()) and float(actual.abs().max()) > 0)
        params = sum(w.numel() for w in weights)
        expected_params = 2 * hidden**2 + 2 * hidden * kv_heads * head_dim
        check(prefix + 'parameter_formula', params == expected_params)
        key_cache = torch.empty(batch, layers, time, kv_heads, head_dim, dtype=dtype)
        value_cache = torch.empty_like(key_cache)
        tensor_bytes = sum(a.numel() * a.element_size() for a in (key_cache, value_cache))
        formula_bytes = 2 * batch * layers * time * kv_heads * head_dim * element_bytes
        check(prefix + 'cache_formula_matches_payload', tensor_bytes == formula_bytes)
        check(prefix + 'cache_ratio', tensor_bytes * query_heads == mha_cache * kv_heads)
        cut = time // 2
        with torch.no_grad():
            changed = x.detach().clone()
            changed[:, cut + 1:] += 3.0 * torch.randn_like(changed[:, cut + 1:])
            changed_out = attention(changed, *weights, query_heads, kv_heads)
            prefix_error = float((out[:, :cut + 1] - changed_out[:, :cut + 1]).abs().max())
            check(prefix + 'causal_prefix_unchanged', prefix_error < 1e-10)
            check(prefix + 'perturbation_has_effect', float((out[:, -1] - changed_out[:, -1]).abs().max()) > 1e-8)
            wrong_delta = None
            if kv_heads > 1 and repeats > 1:
                wrong = attention(x, *weights, query_heads, kv_heads, mode='wrong_repeat')
                wrong_delta = float((out - wrong).abs().max())
                check(prefix + 'wrong_repeat_negative_control', wrong_delta > 1e-7)
        rows.append({'kv_heads': kv_heads, 'query_heads': query_heads, 'query_per_group': repeats,
                     'query_to_kv': [h // repeats for h in range(query_heads)],
                     'parameter_count': params, 'kv_projection_parameters': 2 * hidden * kv_heads * head_dim,
                     'cache_payload_bytes': tensor_bytes, 'cache_fraction_of_mha': tensor_bytes / mha_cache,
                     'forward_max_abs_error': forward_error, 'gradient_max_abs_errors': gradient_errors,
                     'causal_prefix_max_abs_error': prefix_error, 'wrong_repeat_max_abs_delta': wrong_delta})
    check('mha_parameters_equal_four_square_projections', rows[-1]['parameter_count'] == 4 * hidden**2)
    check('cache_monotonic_with_kv_heads', all(a['cache_payload_bytes'] < b['cache_payload_bytes']
                                             for a, b in zip(rows, rows[1:])))
    return {'chapter': '07-gqa', 'seed': seed, 'checks': checks,
            'metrics': {'mode': 'quick' if quick else 'standard', 'dtype': str(dtype),
                        'batch': batch, 'sequence_length': time, 'hidden_size': hidden,
                        'head_dim': head_dim, 'cache_layers': layers, 'element_bytes': element_bytes,
                        'cases': rows, 'memory_scope': 'tensor payload only; not allocator or process peak'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args.seed, args.quick)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')


if __name__ == '__main__':
    main()
