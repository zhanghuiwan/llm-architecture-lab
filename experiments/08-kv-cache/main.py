#!/usr/bin/env python3
"""Independent CPU experiment: two-layer RoPE + GQA decoding with a KV cache."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn


def rms_norm(x, weight):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight


def rope(x, positions, base=10000.0):
    """Adjacent even/odd pairing; x is [B, H, T, dh]."""
    dh = x.shape[-1]
    frequencies = base ** (-torch.arange(0, dh, 2, dtype=x.dtype, device=x.device) / dh)
    angles = positions.to(x.dtype)[:, None] * frequencies[None, :]
    c, s = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width, query_heads, kv_heads, head_dim):
        super().__init__()
        self.hq, self.hkv, self.dh = query_heads, kv_heads, head_dim
        self.attention_norm = nn.Parameter(torch.ones(width))
        self.ffn_norm = nn.Parameter(torch.ones(width))
        self.wq = nn.Linear(width, query_heads * head_dim, bias=False)
        self.wk = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(query_heads * head_dim, width, bias=False)
        self.gate = nn.Linear(width, 2 * width, bias=False)
        self.up = nn.Linear(width, 2 * width, bias=False)
        self.down = nn.Linear(2 * width, width, bias=False)

    def forward(self, x, previous=None, wrong_rope=False, wrong_mask=False):
        b, count, width = x.shape
        past = 0 if previous is None else previous[0].shape[2]
        z = rms_norm(x, self.attention_norm)
        q = self.wq(z).reshape(b, count, self.hq, self.dh).transpose(1, 2)
        k = self.wk(z).reshape(b, count, self.hkv, self.dh).transpose(1, 2)
        v = self.wv(z).reshape(b, count, self.hkv, self.dh).transpose(1, 2)
        offset = 0 if wrong_rope else past
        positions = torch.arange(offset, offset + count, device=x.device)
        q, k = rope(q, positions), rope(k, positions)
        if previous is not None:
            k = torch.cat((previous[0], k), dim=2)
            v = torch.cat((previous[1], v), dim=2)
        total = k.shape[2]
        if wrong_mask:
            # Deliberately wrong: upper-left triangle ignores the cache offset.
            allowed = torch.ones(count, total, dtype=torch.bool, device=x.device).tril()
        else:
            allowed = torch.arange(total, device=x.device)[None, :] <= (
                past + torch.arange(count, device=x.device)[:, None]
            )
        repeats = self.hq // self.hkv
        # Expansion is only the readable reference computation, not stored cache.
        keys = k.repeat_interleave(repeats, dim=1)
        values = v.repeat_interleave(repeats, dim=1)
        scores = q @ keys.transpose(-2, -1) / self.dh ** 0.5
        probabilities = scores.masked_fill(~allowed[None, None], -torch.inf).softmax(-1)
        attended = probabilities @ values
        x = x + self.wo(attended.transpose(1, 2).reshape(b, count, self.hq * self.dh))
        z = rms_norm(x, self.ffn_norm)
        x = x + self.down(torch.nn.functional.silu(self.gate(z)) * self.up(z))
        return x, (k, v)


class TinyDecoder(nn.Module):
    def __init__(self, vocab=31, width=24, query_heads=4, kv_heads=2, head_dim=6,
                 layers=2, max_context=64):
        super().__init__()
        self.vocab, self.width = vocab, width
        self.hq, self.hkv, self.dh = query_heads, kv_heads, head_dim
        self.max_context = max_context
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([Block(width, query_heads, kv_heads, head_dim) for _ in range(layers)])
        self.final_norm = nn.Parameter(torch.ones(width))
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def validate_cache(self, ids, cache):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('ids must be nonempty [B,T] without padding')
        if cache is None:
            past = 0
        else:
            if len(cache) != len(self.blocks):
                raise ValueError('cache must contain one K/V pair per layer')
            lengths = []
            for key, value in cache:
                expected_prefix = (ids.shape[0], self.hkv)
                if (key.ndim != 4 or value.shape != key.shape
                        or tuple(key.shape[:2]) != expected_prefix or key.shape[-1] != self.dh):
                    raise ValueError('cache shape mismatch')
                if key.dtype != self.embedding.weight.dtype or value.dtype != key.dtype:
                    raise ValueError('cache dtype mismatch')
                if key.device != ids.device or value.device != ids.device:
                    raise ValueError('cache device mismatch')
                lengths.append(key.shape[2])
            if len(set(lengths)) != 1:
                raise ValueError('cache layers must have equal logical lengths')
            past = lengths[0]
        if past + ids.shape[1] > self.max_context:
            raise ValueError('requested positions exceed the experiment context limit')
        return past

    def forward(self, ids, cache=None, wrong_rope=False, wrong_mask=False):
        self.validate_cache(ids, cache)
        x = self.embedding(ids)
        hidden, updated = [], []
        for index, block in enumerate(self.blocks):
            previous = None if cache is None else cache[index]
            x, pair = block(x, previous, wrong_rope, wrong_mask)
            hidden.append(x)
            updated.append(pair)
        x = rms_norm(x, self.final_norm)
        return self.lm_head(x), hidden, updated


def decode_chunks(model, ids, sizes, **kwargs):
    cache, start = None, 0
    outputs, hidden = [], [[] for _ in model.blocks]
    for size in sizes:
        logits, states, cache = model(ids[:, start:start + size], cache, **kwargs)
        outputs.append(logits)
        for layer, state in enumerate(states):
            hidden[layer].append(state)
        start += size
    if start != ids.shape[1]:
        raise ValueError('chunk sizes must exactly cover the supplied input')
    return torch.cat(outputs, 1), [torch.cat(parts, 1) for parts in hidden], cache


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = TinyDecoder().double().eval()
    batch, length, prompt = (2, 13, 3) if quick else (3, 19, 5)
    ids = torch.randint(0, model.vocab, (batch, length))
    chunks = [4, 3, 1, 5] if quick else [6, 2, 4, 1, 6]
    checks, errors = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    def equivalent(name, actual, reference):
        error = float((actual - reference).abs().max())
        errors[name] = error
        check(name, torch.allclose(actual, reference, atol=1e-10, rtol=1e-9))

    with torch.inference_mode():
        full_logits, full_hidden, full_cache = model(ids)
        single_logits, single_hidden, single_cache = decode_chunks(
            model, ids, [prompt] + [1] * (length - prompt))
        chunk_logits, chunk_hidden, chunk_cache = decode_chunks(model, ids, chunks)
        equivalent('single_step_logits_equal_full', single_logits, full_logits)
        equivalent('chunked_logits_equal_full', chunk_logits, full_logits)
        for layer in range(len(model.blocks)):
            equivalent(f'single_step_layer_{layer}_hidden_equal', single_hidden[layer], full_hidden[layer])
            equivalent(f'chunked_layer_{layer}_hidden_equal', chunk_hidden[layer], full_hidden[layer])
            for index, kind in enumerate(('keys', 'values')):
                equivalent(f'single_step_layer_{layer}_{kind}_equal', single_cache[layer][index], full_cache[layer][index])
                equivalent(f'chunked_layer_{layer}_{kind}_equal', chunk_cache[layer][index], full_cache[layer][index])
        empty = [(torch.empty(batch, model.hkv, 0, model.dh, dtype=torch.float64),
                  torch.empty(batch, model.hkv, 0, model.dh, dtype=torch.float64)) for _ in model.blocks]
        empty_logits, _, empty_cache = model(ids, empty)
        equivalent('explicit_empty_cache_equals_none', empty_logits, full_logits)
        _, _, pref_cache = model(ids[:, :prompt])
        snapshots = [(k.clone(), v.clone()) for k, v in pref_cache]
        next_logits, _, appended = model(ids[:, prompt:prompt + 2], pref_cache)
        equivalent('offset_two_token_chunk_logits_equal', next_logits, full_logits[:, prompt:prompt + 2])
        check('input_cache_not_mutated', all(torch.equal(a, b) for p, q in zip(pref_cache, snapshots) for a, b in zip(p, q)))
        check('append_keeps_old_prefix', all(torch.equal(old, new[:, :, :prompt]) for p, q in zip(snapshots, appended) for old, new in zip(p, q)))
        check('cache_has_one_pair_per_layer', len(chunk_cache) == 2)
        check('cache_stores_kv_heads_not_query_heads', all(k.shape == (batch, model.hkv, length, model.dh) and v.shape == k.shape for k, v in chunk_cache))
        expected = len(model.blocks) * 2 * batch * model.hkv * length * model.dh * 8
        actual = cache_bytes(chunk_cache)
        check('actual_cache_tensor_bytes_equal_formula', actual == expected)
        expanded_budget = len(model.blocks) * 2 * batch * model.hq * length * model.dh * 8
        check('gqa_cache_budget_ratio', expanded_budget == actual * (model.hq // model.hkv))
        check('full_cache_finite', all(torch.isfinite(t).all() for pair in full_cache for t in pair))
        future_changed = ids.clone()
        future_changed[:, prompt:] = (future_changed[:, prompt:] + 1) % model.vocab
        changed_logits, _, _ = model(future_changed)
        equivalent('future_tokens_do_not_change_prefix', changed_logits[:, :prompt], full_logits[:, :prompt])
        wrong_position, _, _ = decode_chunks(model, ids, [prompt] + [1] * (length - prompt), wrong_rope=True)
        wrong_triangle, _, _ = decode_chunks(model, ids, chunks, wrong_mask=True)
        position_error = float((wrong_position[:, prompt:] - full_logits[:, prompt:]).abs().max())
        mask_error = float((wrong_triangle - full_logits).abs().max())
        check('reset_rope_positions_negative_control_detected', position_error > 1e-5)
        check('upper_left_triangle_negative_control_detected', mask_error > 1e-5)
        # A cache carries context. The same suffix after unrelated context is a different request.
        suffix = ids[:, prompt:prompt + 2]
        fresh_logits, _, _ = model(suffix)
        stale_logits, _, _ = model(suffix, pref_cache)
        stale_error = float((fresh_logits - stale_logits).abs().max())
        check('stale_request_cache_changes_logits', stale_error > 1e-5)
        reset_logits, _, _ = model(suffix, cache=None)
        equivalent('reset_cache_recovers_fresh_request', reset_logits, fresh_logits)
        # Cache validation can catch structure errors, but not context ownership errors.
        malformed = [(pref_cache[0][0][:1], pref_cache[0][1][:1]), pref_cache[1]]
        bad_lengths = [pref_cache[0], (pref_cache[1][0][:, :, :-1], pref_cache[1][1][:, :, :-1])]
        invalid_cases = {
            'wrong_batch_cache_rejected': lambda: model(ids[:, :1], malformed),
            'unequal_layer_lengths_rejected': lambda: model(ids[:, :1], bad_lengths),
            'missing_layer_cache_rejected': lambda: model(ids[:, :1], pref_cache[:1]),
            'empty_input_rejected': lambda: model(ids[:, :0]),
            'context_limit_rejected': lambda: model(torch.zeros(batch, 65, dtype=torch.long)),
        }
        for name, action in invalid_cases.items():
            raised = False
            try:
                action()
            except ValueError:
                raised = True
            check(name, raised)
    return {'chapter': '08-kv-cache', 'seed': seed, 'checks': checks, 'metrics': {
        'mode': 'quick' if quick else 'standard', 'dtype': 'float64', 'batch': batch,
        'sequence_length': length, 'prefill_length': prompt, 'chunk_sizes': chunks,
        'layers': 2, 'query_heads': model.hq, 'kv_heads': model.hkv, 'head_dim': model.dh,
        'maximum_absolute_errors': errors, 'wrong_rope_max_error': position_error,
        'wrong_mask_max_error': mask_error, 'stale_request_max_error': stale_error,
        'cache_tensor_bytes': actual, 'cache_formula_bytes': expected,
        'mha_head_count_budget_bytes': expanded_budget,
        'cache_to_mha_head_budget_ratio': actual / expanded_budget,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'checks_passed': len(checks),
        'scope': 'CPU FP64 random model, equal-length unpadded batch, contiguous append-only cache; no training or latency benchmark',
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args.seed, args.quick)
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
