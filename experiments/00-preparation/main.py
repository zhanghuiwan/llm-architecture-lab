#!/usr/bin/env python3
"""第 00 章：CPU 上的形状、广播、数值稳定性与梯度实验。"""
import argparse
import json
from pathlib import Path

import torch


def run(seed: int, quick: bool) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    b, t, d, h = (2, 5, 12, 3) if quick else (3, 7, 20, 4)
    dh = d // h
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    x = torch.randn(b, t, d, dtype=dtype)
    w = torch.randn(d, d, dtype=dtype) / d**0.5
    projected = x @ w
    q = projected.reshape(b, t, h, dh).transpose(1, 2)
    check("head_dimension_is_not_sequence_length", dh != t)
    check("split_shape", tuple(q.shape) == (b, h, t, dh))
    check("projection_matches_einsum", torch.allclose(projected, torch.einsum("btd,df->btf", x, w)))
    joined = q.transpose(1, 2).contiguous().view(b, t, d)
    check("split_join_preserves_values", torch.equal(joined, projected))
    check("transpose_is_noncontiguous", not q.is_contiguous())
    view_failed = False
    try:
        q.view(b, h * t, dh)
    except RuntimeError:
        view_failed = True
    check("view_rejects_incompatible_strides", view_failed)
    check("reshape_preserves_logical_order", torch.equal(q.reshape(b, h * t, dh), q.contiguous().view(b, h * t, dh)))

    k, v = torch.randn_like(q), torch.randn_like(q)
    scores = q @ k.transpose(-2, -1) / dh**0.5
    check("scores_shape", tuple(scores.shape) == (b, h, t, t))
    allowed = torch.ones(t, t, dtype=torch.bool).tril()
    masked = scores.masked_fill(~allowed[None, None, :, :], -torch.inf)
    probabilities = masked.softmax(dim=-1)
    check("softmax_rows_sum_to_one", torch.allclose(probabilities.sum(-1), torch.ones(b, h, t, dtype=dtype)))
    check("causal_future_probabilities_zero", torch.equal(probabilities.masked_select(~allowed[None, None, :, :]), torch.zeros(b * h * t * (t - 1) // 2, dtype=dtype)))
    explicit = torch.empty_like(probabilities)
    for bi in range(b):
        for hi in range(h):
            explicit[bi, hi] = scores[bi, hi].masked_fill(~allowed, -torch.inf).softmax(-1)
    check("broadcast_matches_explicit_batches", torch.allclose(probabilities, explicit))
    out = probabilities @ v
    reference = torch.empty_like(out)
    for position in range(t):
        reference[:, :, position] = (probabilities[:, :, position, :, None] * v).sum(-2)
    check("weighted_sum_matches_explicit_sum", torch.allclose(out, reference))
    wrong = scores.softmax(dim=-2)
    check("wrong_axis_preserves_shape", wrong.shape == scores.shape)
    check("wrong_axis_detected_by_row_sums", not torch.allclose(wrong.sum(-1), torch.ones(b, h, t, dtype=dtype)))

    huge = torch.tensor([1000.0, 1001.0, 999.0], dtype=dtype)
    naive_exp = huge.exp()
    stable_exp = (huge - huge.max()).exp()
    stable = stable_exp / stable_exp.sum()
    check("naive_exponent_overflows", bool(torch.isinf(naive_exp).all()))
    check("stable_softmax_is_finite", bool(torch.isfinite(stable).all()))
    check("stable_matches_torch", torch.allclose(stable, huge.softmax(-1)))
    check("softmax_shift_invariance", torch.allclose(huge.softmax(-1), (huge - 1000).softmax(-1)))
    all_masked = torch.full((t,), -torch.inf, dtype=dtype).softmax(-1)
    check("all_masked_row_is_invalid", bool(torch.isnan(all_masked).all()))

    # 共享线性投影的光滑标量目标：可逐坐标做有限差分。
    xg = torch.randn(b, t, d, dtype=dtype, requires_grad=True)
    wg = (torch.randn(d, 3, dtype=dtype) / d**0.5).requires_grad_()
    loss = (xg @ wg).square().mean()
    loss.backward()
    check("input_gradient_shape", xg.grad.shape == xg.shape)
    check("weight_gradient_shape", wg.grad.shape == wg.shape)
    check("gradient_finite", bool(torch.isfinite(xg.grad).all() and torch.isfinite(wg.grad).all()))
    coordinates = [(0, 0), (d - 1, 2), (d // 2, 1)]
    epsilon = 1e-6
    errors = []
    with torch.no_grad():
        for row, column in coordinates:
            wp, wm = wg.clone(), wg.clone()
            wp[row, column] += epsilon
            wm[row, column] -= epsilon
            numerical = ((xg @ wp).square().mean() - (xg @ wm).square().mean()) / (2 * epsilon)
            errors.append(float((numerical - wg.grad[row, column]).abs()))
    check("finite_difference_matches_autograd", max(errors) < 1e-7)
    first_grad = wg.grad.clone()
    (xg @ wg).square().mean().backward()
    check("backward_accumulates_gradients", torch.allclose(wg.grad, 2 * first_grad))

    layer = torch.nn.Linear(d, 3, dtype=dtype)
    layer.eval()
    check("eval_still_records_gradients", layer(x).requires_grad)
    with torch.no_grad():
        no_grad_out = layer(x)
    check("no_grad_disables_recording", not no_grad_out.requires_grad)
    dropout = torch.nn.Dropout(0.5)
    ones = torch.ones(1024, dtype=dtype)
    dropout.train()
    train_out = dropout(ones)
    check("training_dropout_changes_values", bool((train_out != ones).any()))
    dropout.eval()
    check("eval_dropout_is_identity", torch.equal(dropout(ones), ones))

    return {
        "chapter": "00-preparation",
        "seed": seed,
        "checks": checks,
        "metrics": {
            "mode": "quick" if quick else "standard",
            "dtype": str(dtype),
            "input_shape": [b, t, d],
            "head_shape": list(q.shape),
            "head_strides": list(q.stride()),
            "score_shape": list(scores.shape),
            "loss": float(loss.detach()),
            "finite_difference_coordinates": [list(c) for c in coordinates],
            "finite_difference_max_abs_error": max(errors),
            "row_sum_max_abs_error": float((probabilities.sum(-1) - 1).abs().max()),
            "stable_softmax": stable.tolist(),
            "checks_passed": len(checks),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.seed, args.quick)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
