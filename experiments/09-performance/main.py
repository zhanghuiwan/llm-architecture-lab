#!/usr/bin/env python3
"""第09章：在 CPU 上核对注意力语义，并测量两个固定工作负载。"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch
import torch.nn.functional as F


def manual_attention(q, k, v, allowed):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
    return probabilities @ v


def sdpa_attention(q, k, v, allowed):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                         dropout_p=0.0, is_causal=False)


def benchmark(function, warmup, repeats, inner_calls):
    # 输入已经生成，no_grad 的上下文也在计时之外。CPU 调用在这里同步完成。
    with torch.no_grad():
        for _ in range(warmup):
            function()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(inner_calls):
                function()
            samples.append((time.perf_counter_ns() - start) / inner_calls / 1e6)
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "warmup_calls": warmup,
        "rounds": repeats,
        "inner_calls": inner_calls,
    }


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    b, h, t, d = (2, 4, 32, 16) if quick else (2, 4, 96, 24)
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    allowed = torch.ones(t, t, dtype=torch.bool).tril()
    q = torch.randn(b, h, t, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    a = manual_attention(q, k, v, allowed)
    ref = sdpa_attention(q, k, v, allowed)
    check("prefill_shape", tuple(a.shape) == (b, h, t, d))
    prefill_error = float((a - ref).abs().max().detach())
    check("float64_prefill_forward_matches", prefill_error < 1e-10)
    upstream = torch.randn_like(a)
    g_manual = torch.autograd.grad(a, (q, k, v), upstream)
    g_sdpa = torch.autograd.grad(ref, (q, k, v), upstream)
    gradient_errors = {}
    for name, first, second in zip(("q", "k", "v"), g_manual, g_sdpa):
        error = float((first - second).abs().max())
        gradient_errors[name] = error
        check(f"float64_{name}_gradient_matches", error < 1e-10)
        check(f"{name}_gradient_is_finite", bool(torch.isfinite(first).all()))

    # 最后一个 Query 可以读取全部 t 个 Key；is_causal=True 的非方阵语义不同。
    q_decode = q.detach()[..., -1:, :].contiguous()
    decode_mask = torch.ones(1, t, dtype=torch.bool)
    full_last = a.detach()[..., -1:, :]
    dec_manual = manual_attention(q_decode, k.detach(), v.detach(), decode_mask)
    dec_sdpa = sdpa_attention(q_decode, k.detach(), v.detach(), decode_mask)
    decode_error = float((dec_manual - dec_sdpa).abs().max())
    check("decode_shape", tuple(dec_manual.shape) == (b, h, 1, d))
    check("decode_matches_last_prefill_position", torch.allclose(dec_manual, full_last, atol=1e-10, rtol=1e-10))
    check("float64_decode_forward_matches", decode_error < 1e-10)
    wrong_decode = F.scaled_dot_product_attention(q_decode, k.detach(), v.detach(),
                                                dropout_p=0.0, is_causal=True)
    negative_difference = float((wrong_decode - dec_sdpa).abs().max())
    check("wrong_non_square_causal_alignment_is_detected", negative_difference > 1e-6)

    # 计时使用常见 FP32，正确性与计时的精度分别记录。
    q32, k32, v32 = (x.detach().to(torch.float32).contiguous() for x in (q, k, v))
    qd32 = q32[..., -1:, :].contiguous()
    fp32_error = float((manual_attention(q32, k32, v32, allowed) -
                        sdpa_attention(q32, k32, v32, allowed)).abs().max())
    check("timed_float32_prefill_semantics_match", fp32_error < 1e-5)
    decode_fp32_error = float((manual_attention(qd32, k32, v32, decode_mask) -
                               sdpa_attention(qd32, k32, v32, decode_mask)).abs().max())
    check("timed_float32_decode_semantics_match", decode_fp32_error < 1e-5)
    warmup, repeats, inner = (3, 9, 3) if quick else (8, 25, 5)
    workloads = {"prefill": (q32, allowed), "decode": (qd32, decode_mask)}
    timing = {}
    for workload, (queries, mask) in workloads.items():
        timing[workload] = {}
        for name, implementation in (("manual", manual_attention), ("sdpa", sdpa_attention)):
            result = benchmark(lambda: implementation(queries, k32, v32, mask),
                               warmup, repeats, inner)
            timing[workload][name] = result
            check(f"{workload}_{name}_samples_are_positive_and_finite",
                  len(result["samples_ms"]) == repeats and
                  all(math.isfinite(value) and value > 0 for value in result["samples_ms"]))
        # 仅报告观测比值，不将哪一个更快作为通过条件。
        timing[workload]["manual_over_sdpa_median_ratio"] = (
            timing[workload]["manual"]["median_ms"] / timing[workload]["sdpa"]["median_ms"])
    item_bytes = q32.element_size()
    theory = {}
    for workload, (queries, _) in workloads.items():
        length = queries.shape[-2]
        theory[workload] = {
            "query_length": length, "key_length": t,
            "dense_matmul_flops": 4 * b * h * length * t * d,
            "one_dense_score_tensor_bytes": b * h * length * t * item_bytes,
            "kv_tensor_bytes": 2 * b * h * t * d * item_bytes,
        }
    check("kv_bytes_match_stored_k_and_v", theory["decode"]["kv_tensor_bytes"] ==
          k32.numel() * k32.element_size() + v32.numel() * v32.element_size())
    check("decode_theoretical_matmul_flops_are_prefill_divided_by_t",
          theory["prefill"]["dense_matmul_flops"] == t * theory["decode"]["dense_matmul_flops"])
    return {
        "chapter": "09-performance", "seed": seed, "checks": checks,
        "metrics": {
            "mode": "quick" if quick else "standard",
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "platform": platform.platform(), "machine": platform.machine(),
                            "device": "cpu", "torch_threads": torch.get_num_threads(),
                            "logical_cpu_count": os.cpu_count()},
            "shapes": {"prefill_q": list(q32.shape), "decode_q": list(qd32.shape),
                       "k": list(k32.shape), "v": list(v32.shape)},
            "correctness_dtype": "float64", "timing_dtype": "float32",
            "prefill_forward_max_abs_error": prefill_error,
            "gradient_max_abs_errors": gradient_errors,
            "decode_forward_max_abs_error": decode_error,
            "float32_prefill_max_abs_error": fp32_error,
            "float32_decode_max_abs_error": decode_fp32_error,
            "wrong_decode_mask_max_abs_difference": negative_difference,
            "theory": theory, "timing": timing,
            "timing_scope": "CPU attention forward only; existing tensors and mask; no_grad; no projections/cache append/tokenizer/sampling/backward",
            "cuda_executed": False, "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None, "cpu_process_peak_memory_bytes": None,
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
