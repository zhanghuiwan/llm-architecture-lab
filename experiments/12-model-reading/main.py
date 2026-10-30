#!/usr/bin/env python3
"""固定源码概念的独立教学核对，不加载官方权重。"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DTYPE = torch.float64


def rms(x, weight=None, eps=1e-6):
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def rope(x, positions):
    """Split-half coordinate pairing, with sequence at axis -2."""
    d = x.shape[-1]
    assert d % 2 == 0
    frequency = 10000.0 ** (-torch.arange(0, d, 2, dtype=DTYPE) / d)
    angle = positions.to(DTYPE)[:, None] * frequency[None, :]
    phase = torch.cat((angle, angle), dim=-1)
    rotated = torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)
    return x * phase.cos() + rotated * phase.sin()


class TinyQwenBlock(nn.Module):
    def __init__(self, hidden=20, heads=4, kv_heads=2, head_dim=6, intermediate=32):
        super().__init__()
        assert heads % kv_heads == 0 and head_dim % 2 == 0
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.input_norm = nn.Parameter(torch.ones(hidden, dtype=DTYPE))
        self.post_norm = nn.Parameter(torch.ones(hidden, dtype=DTYPE))
        self.q_norm = nn.Parameter(torch.ones(head_dim, dtype=DTYPE))
        self.k_norm = nn.Parameter(torch.ones(head_dim, dtype=DTYPE))
        def linear(a, b):
            return nn.Linear(a, b, bias=False, dtype=DTYPE)
        self.q_proj = linear(hidden, heads * head_dim)
        self.k_proj = linear(hidden, kv_heads * head_dim)
        self.v_proj = linear(hidden, kv_heads * head_dim)
        self.o_proj = linear(heads * head_dim, hidden)
        self.gate_proj = linear(hidden, intermediate)
        self.up_proj = linear(hidden, intermediate)
        self.down_proj = linear(intermediate, hidden)

    def forward(self, x, reference=False, wrong_norm=False):
        b, t, _ = x.shape
        z = rms(x, self.input_norm)
        q = self.q_proj(z).reshape(b, t, self.heads, self.head_dim)
        k = self.k_proj(z).reshape(b, t, self.kv_heads, self.head_dim)
        if wrong_norm:
            q = q * torch.rsqrt(q.square().mean((-2, -1), keepdim=True) + 1e-6) * self.q_norm
            k = k * torch.rsqrt(k.square().mean((-2, -1), keepdim=True) + 1e-6) * self.k_norm
        else:
            q, k = rms(q, self.q_norm), rms(k, self.k_norm)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        positions = torch.arange(t)
        q, k = rope(q, positions), rope(k, positions)
        v = self.v_proj(z).reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        group = self.heads // self.kv_heads
        if reference:
            repeated_k = k.repeat_interleave(group, dim=1)
            repeated_v = v.repeat_interleave(group, dim=1)
            y = F.scaled_dot_product_attention(q, repeated_k, repeated_v, is_causal=True)
        else:
            grouped_q = q.reshape(b, self.kv_heads, group, t, self.head_dim)
            scores = torch.einsum('bhgtd,bhsd->bhgts', grouped_q, k) / math.sqrt(self.head_dim)
            allowed = torch.ones(t, t, dtype=torch.bool).tril()
            probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
            y = torch.einsum('bhgts,bhsv->bhgtv', probabilities, v).reshape(b, self.heads, t, self.head_dim)
        y = self.o_proj(y.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))
        x = x + y
        z = rms(x, self.post_norm)
        return x + self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))


def mla(parameters, heads, dn, dr, dv, absorbed):
    x, w_c, w_kp, w_qn, w_qp, u_k, u_v, w_o = parameters
    b, t, _ = x.shape
    c = rms(x @ w_c)
    kp = rope((x @ w_kp)[:, None], torch.arange(t))[:, 0]
    qn = (x @ w_qn).reshape(b, t, heads, dn).transpose(1, 2)
    qp = rope((x @ w_qp).reshape(b, t, heads, dr).transpose(1, 2), torch.arange(t))
    if absorbed:
        transformed_q = torch.einsum('bhtd,hrd->bhtr', qn, u_k)
        scores = (torch.einsum('bhtr,bsr->bhts', transformed_q, c)
                  + torch.einsum('bhtd,bsd->bhts', qp, kp)) / math.sqrt(dn + dr)
        cache_tensors = (c, kp)
    else:
        kn = torch.einsum('btr,hrd->bhtd', c, u_k)
        values = torch.einsum('btr,hrd->bhtd', c, u_v)
        expanded_k = torch.cat((kn, kp[:, None].expand(-1, heads, -1, -1)), -1)
        expanded_q = torch.cat((qn, qp), -1)
        scores = (expanded_q @ expanded_k.transpose(-2, -1)) / math.sqrt(dn + dr)
        cache_tensors = (expanded_k, values)
    allowed = torch.ones(t, t, dtype=torch.bool).tril()
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    if absorbed:
        latent_output = torch.einsum('bhts,bsr->bhtr', weights, c)
        y = torch.einsum('bhtr,hrd->bhtd', latent_output, u_v)
    else:
        y = weights @ values
    output = y.transpose(1, 2).reshape(b, t, heads * dv) @ w_o
    return output, scores, weights, cache_tensors


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)
    def error(a, b):
        return float((a.detach() - b.detach()).abs().max())

    b, t, hidden, heads, kv_heads, dh, intermediate = (2, 5, 20, 4, 2, 6, 32) if quick else (2, 7, 30, 6, 2, 8, 48)
    block = TinyQwenBlock(hidden, heads, kv_heads, dh, intermediate)
    reference = copy.deepcopy(block)
    x = torch.randn(b, t, hidden, dtype=DTYPE, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    output, reference_output = block(x), reference(xr, reference=True)
    check('qwen_head_dim_independent_of_hidden_div_heads', dh != hidden // heads)
    check('qwen_q_projection_shape', tuple(block.q_proj.weight.shape) == (heads * dh, hidden))
    check('qwen_o_projection_shape', tuple(block.o_proj.weight.shape) == (hidden, heads * dh))
    check('qwen_per_head_qk_norm_weights', block.q_norm.numel() == dh and block.k_norm.numel() == dh)
    check('qwen_output_shape', output.shape == x.shape)
    forward_error = error(output, reference_output)
    check('qwen_forward_matches_sdpa', forward_error < 1e-10)
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    (reference_output * probe).sum().backward()
    input_error = error(x.grad, xr.grad)
    check('qwen_input_gradient_matches', input_error < 1e-9)
    gradient_errors = {}
    for (name, p), (_, pr) in zip(block.named_parameters(), reference.named_parameters()):
        gradient_errors[name] = error(p.grad, pr.grad)
        check('qwen_gradient_' + name, gradient_errors[name] < 1e-9)
    check('qwen_all_gradients_finite', bool(all(torch.isfinite(p.grad).all() for p in block.parameters())))
    expected_parameters = (2 * hidden + 2 * dh + 2 * hidden * heads * dh
                           + 2 * hidden * kv_heads * dh + 3 * hidden * intermediate)
    actual_parameters = sum(p.numel() for p in block.parameters())
    check('qwen_parameter_count', actual_parameters == expected_parameters)
    with torch.no_grad():
        perturbed = x.detach().clone()
        perturbed[:, -1] += 100
        prefix_error = error(block(perturbed)[:, :-1], output[:, :-1])
        check('qwen_causal_prefix_unchanged', prefix_error < 1e-10)
        wrong_norm_error = error(block(x.detach(), wrong_norm=True), output)
        check('qwen_wrong_global_qk_norm_detected', wrong_norm_error > 1e-6)

    rank, dn, dr, dv = (5, 6, 4, 6) if quick else (7, 8, 4, 6)
    shapes = [(b, t, hidden), (hidden, rank), (hidden, dr),
              (hidden, heads * dn), (hidden, heads * dr),
              (heads, rank, dn), (heads, rank, dv), (heads * dv, hidden)]
    parameters = [((torch.randn(shape, dtype=DTYPE) / math.sqrt(shape[-2] if len(shape) == 2 else shape[-1]))
                   .requires_grad_()) for shape in shapes]
    clones = [p.detach().clone().requires_grad_() for p in parameters]
    expanded, scores_e, weights_e, cache_e = mla(parameters, heads, dn, dr, dv, False)
    absorbed, scores_a, weights_a, cache_a = mla(clones, heads, dn, dr, dv, True)
    mla_forward_error = error(expanded, absorbed)
    score_error, weight_error = error(scores_e, scores_a), error(weights_e, weights_a)
    check('mla_scores_equal', score_error < 1e-10)
    check('mla_probabilities_equal', weight_error < 1e-10)
    check('mla_output_equal', mla_forward_error < 1e-10)
    probe = torch.randn_like(expanded)
    (expanded * probe).sum().backward()
    (absorbed * probe).sum().backward()
    mla_gradient_errors = [error(p.grad, p2.grad) for p, p2 in zip(parameters, clones)]
    for i, e in enumerate(mla_gradient_errors):
        check('mla_gradient_' + str(i), e < 1e-8)
    expanded_bytes = sum(v.numel() * v.element_size() for v in cache_e)
    latent_bytes = sum(v.numel() * v.element_size() for v in cache_a)
    check('mla_expanded_cache_shape_bytes', expanded_bytes == b * t * heads * (dn + dr + dv) * 8)
    check('mla_latent_cache_shape_bytes', latent_bytes == b * t * (rank + dr) * 8)
    check('mla_selected_example_cache_smaller', latent_bytes < expanded_bytes)
    check('mla_causal_future_weight_zero', bool((weights_a.masked_select(~torch.ones(t, t, dtype=torch.bool).tril()) == 0).all()))
    return {'chapter': '12-model-reading', 'seed': seed, 'checks': checks, 'metrics': {
        'mode': 'quick' if quick else 'standard', 'dtype': str(DTYPE),
        'qwen_teaching_config': {'batch': b, 'sequence': t, 'hidden': hidden, 'heads': heads,
                                 'kv_heads': kv_heads, 'head_dim': dh, 'intermediate': intermediate},
        'qwen_parameters': actual_parameters, 'qwen_forward_max_error': forward_error,
        'qwen_input_gradient_max_error': input_error,
        'qwen_parameter_gradient_max_error': max(gradient_errors.values()),
        'qwen_causal_prefix_error': prefix_error, 'wrong_qk_norm_output_difference': wrong_norm_error,
        'mla_teaching_config': {'rank': rank, 'nope_head_dim': dn, 'rope_head_dim': dr, 'v_head_dim': dv},
        'mla_scores_max_error': score_error, 'mla_probabilities_max_error': weight_error,
        'mla_output_max_error': mla_forward_error, 'mla_gradient_max_error': max(mla_gradient_errors),
        'expanded_cache_tensor_bytes': expanded_bytes, 'latent_cache_tensor_bytes': latent_bytes,
        'cache_payload_ratio': expanded_bytes / latent_bytes,
        'scope': 'FP64 teaching algebra; not official weights, FP8 kernels, YaRN or distributed execution',
    }}


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
