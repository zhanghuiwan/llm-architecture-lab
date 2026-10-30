#!/usr/bin/env python3
"""自包含 CPU bigram 语言建模实验；无需下载数据或导入其他章节。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TRAIN = [
    "春天来了。", "春天风轻。", "春天花开。", "夏天来了。", "夏天雨多。",
    "秋天来了。", "秋天风轻。", "冬天来了。", "冬天雪多。",
    "今天读书。", "今天写字。", "读书明理。", "读书。",
]
VALID = ["春天读书。", "秋天花开。", "今天风轻。"]
PAD, BOS, EOS, IGNORE = 0, 1, 2, -100


class Bigram(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size, dtype=torch.float64)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.table(tokens)


def make_batch(lines: list[str], encode: dict[str, int]):
    rows = [[BOS] + [encode[c] for c in line] + [EOS] for line in lines]
    length = max(len(row) - 1 for row in rows)
    x = torch.full((len(rows), length), PAD, dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    for i, row in enumerate(rows):
        x[i, :len(row)-1] = torch.tensor(row[:-1])
        y[i, :len(row)-1] = torch.tensor(row[1:])
    return x, y, rows


def loss(logits: torch.Tensor, labels: torch.Tensor, reduction="mean"):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=IGNORE,
                           reduction=reduction)


@torch.no_grad()
def generate(model: Bigram, encode, decode, prompt: str, temperature: float,
             seed: int, max_new_tokens=24):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not prompt:
        ids = [BOS]
    else:
        ids = [BOS] + [encode[c] for c in prompt]
    generator = torch.Generator().manual_seed(seed)
    emitted = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids[-1]])).squeeze(0) / temperature
        logits[PAD] = -torch.inf
        logits[BOS] = -torch.inf
        token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        emitted.append(token)
        if token == EOS:
            break
        ids.append(token)
    text = prompt + "".join(decode[t] for t in emitted if t != EOS)
    return text, emitted


def run(quick: bool, seed: int):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    alphabet = sorted(set("".join(TRAIN)))
    encode = {c: i + 3 for i, c in enumerate(alphabet)}
    decode = {v: k for k, v in encode.items()}
    assert all(c in encode for c in "".join(VALID))
    vocab_size = len(encode) + 3
    x, y, rows = make_batch(TRAIN, encode)
    vx, vy, _ = make_batch(VALID, encode)
    checks = {}

    def check(name, condition):
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            raise AssertionError(name)

    check("input_label_shift", all(
        x[i, :len(row)-1].tolist() == row[:-1]
        and y[i, :len(row)-1].tolist() == row[1:]
        for i, row in enumerate(rows)))
    check("bos_never_target_eos_is_target", not bool((y == BOS).any()) and
          int((y == EOS).sum()) == len(TRAIN))
    check("train_validation_sentences_disjoint", not set(TRAIN) & set(VALID))
    check("padding_present", bool((y == IGNORE).any()))
    model = Bigram(vocab_size)
    logits = model(x)
    check("logits_shape", logits.shape == (*x.shape, vocab_size))
    valid = y != IGNORE
    n_tokens = int(valid.sum())
    manual = -logits.log_softmax(-1)[valid].gather(1, y[valid].unsqueeze(-1)).mean()
    check("cross_entropy_matches_log_softmax", torch.allclose(
        loss(logits, y), manual, atol=1e-12, rtol=1e-12))
    uniform = torch.zeros_like(logits)
    check("uniform_loss_is_log_vocab", abs(float(loss(uniform, y)) - math.log(vocab_size)) < 1e-12)
    check("argmax_of_logits_equals_probabilities", torch.equal(logits.argmax(-1), logits.softmax(-1).argmax(-1)))
    probe = logits.detach().clone().requires_grad_()
    loss(probe, y).backward()
    check("ignored_logits_have_zero_gradient", bool((probe.grad[~valid] == 0).all()))
    changed = logits.detach().clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 100
    check("ignored_logits_do_not_change_loss", torch.equal(loss(changed, y), loss(logits, y)))
    parts = [(x[:1], y[:1]), (x[1:], y[1:])]
    weighted = sum(loss(model(a), b, reduction="sum") for a, b in parts) / n_tokens
    check("token_weighted_microbatches_equal_full_loss", torch.allclose(weighted, loss(logits, y), atol=1e-12))
    # 固定反例：一个 token 的 loss 为 4，九个 token 的 loss 均为 1。
    correct = (4.0 * 1 + 1.0 * 9) / 10
    wrong = (4.0 + 1.0) / 2
    check("unweighted_batch_mean_negative_control", abs(correct - 1.3) < 1e-12 and abs(correct - wrong) > 1)
    counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float64)
    for a, b in zip(x[valid].tolist(), y[valid].tolist()):
        counts[a, b] += 1
    probabilities = counts / counts.sum(-1, keepdim=True).clamp_min(1)
    floor = float(-(counts * probabilities.clamp_min(1e-300).log()).sum() / n_tokens)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    steps = 80 if quick else 240
    initial = float(loss(model(x), y).detach())
    history = [initial]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        current = loss(model(x), y)
        current.backward()
        optimizer.step()
        if (step + 1) % max(1, steps // 4) == 0:
            history.append(float(loss(model(x), y).detach()))
    with torch.no_grad():
        train_loss = float(loss(model(x), y))
        validation_loss = float(loss(model(vx), vy))
        check("loss_improved", train_loss < initial * 0.8)
        check("loss_not_below_empirical_bigram_floor", train_loss >= floor - 1e-10)
        check("fit_close_to_empirical_floor", train_loss - floor < 0.03)
        # 相同当前字符始终产生相同分布：明确模型看不到更早的历史。
        a = model(torch.tensor([[encode["春"], encode["天"]]]))
        b = model(torch.tensor([[encode["冬"], encode["天"]]]))
        check("bigram_has_no_earlier_context", torch.equal(a[:, -1], b[:, -1]))
        base = model(torch.tensor([encode["天"]])).squeeze(0)
        p_cold = (base / 0.6).softmax(-1)
        p_hot = (base / 1.6).softmax(-1)
        entropy = lambda p: float(-(p * p.clamp_min(1e-300).log()).sum())
        check("temperature_changes_entropy", entropy(p_cold) < entropy(p_hot))
    sample, ids = generate(model, encode, decode, "春", 0.8, seed)
    same_sample, same_ids = generate(model, encode, decode, "春", 0.8, seed)
    check("sampling_seed_reproducible", sample == same_sample and ids == same_ids)
    check("sampling_excludes_pad_and_bos", not any(t in (PAD, BOS) for t in ids))
    metrics = {
        "model": "character bigram, not a Transformer", "vocab_size": vocab_size,
        "train_sentences": len(TRAIN), "validation_sentences": len(VALID),
        "train_valid_tokens": n_tokens, "steps": steps,
        "uniform_loss": math.log(vocab_size), "initial_train_loss": initial,
        "final_train_loss": train_loss, "validation_loss": validation_loss,
        "train_perplexity": math.exp(train_loss), "validation_perplexity": math.exp(validation_loss),
        "empirical_train_bigram_floor": floor, "train_loss_history": history,
        "sample_prompt": "春", "sample_temperature": 0.8, "sample": sample,
        "sample_token_ids": ids, "cold_entropy": entropy(p_cold), "hot_entropy": entropy(p_hot),
    }
    return {"chapter": "01-language-modeling", "seed": seed, "checks": checks, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.quick, args.seed)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
