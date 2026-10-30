#!/usr/bin/env python3
"""Independent CPU tiny-GPT experiment; no dataset or weight downloads."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.view(batch, length, self.heads, self.head_dim)
                   .transpose(1, 2) for a in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = torch.ones(length, length, dtype=torch.bool,
                             device=x.device).tril()
        probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        values = (probs @ v).transpose(1, 2).contiguous()
        return self.proj(values.view(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context=24, width=32, heads=4, layers=2,
                 tied=True):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._initialize)
        if tied:
            self.head.weight = self.token.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("expected [B,T] with 1 <= T <= context")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt, count, generator):
        result = prompt.clone()
        for _ in range(count):
            logits = self(result[:, -self.context:])[:, -1]
            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, 1, generator=generator)
            result = torch.cat((result, next_id), dim=1)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    text = "春天读书，夏天看海。秋天写字，冬天听雪。读书看海，写字听雪。" * 8
    chars = sorted(set(text))
    encode = {char: i for i, char in enumerate(chars)}
    ids = torch.tensor([encode[char] for char in text], dtype=torch.long)
    length, batch, width, heads, layers, context = 16, 4, 32, 4, 2, 16
    starts = [0, 5, 13, 21]
    x = torch.stack([ids[start:start + length] for start in starts])
    y = torch.stack([ids[start + 1:start + length + 1] for start in starts])
    vocab_size = len(chars)
    model = TinyGPT(vocab_size, context, width, heads, layers)
    untied = TinyGPT(vocab_size, context, width, heads, layers, tied=False)
    parameters = sum(p.numel() for p in model.parameters())
    formula = vocab_size * width + context * width + layers * (
        12 * width * width + 13 * width) + 2 * width
    check("input_target_shift", torch.equal(x[:, 1:], y[:, :-1]))
    check("embedding_head_same_parameter", model.head.weight is model.token.weight)
    check("exact_parameter_formula", parameters == formula)
    check("untied_adds_vocab_times_width", sum(p.numel() for p in untied.parameters())
          - parameters == vocab_size * width)
    check("tied_parameter_registered_once", sum(p is model.token.weight
          for p in model.parameters()) == 1)
    with torch.no_grad():
        logits = model(x)
        check("logits_shape", logits.shape == (batch, length, vocab_size))
        check("finite_logits", torch.isfinite(logits).all())
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        gathered = -logits.log_softmax(-1).gather(-1, y.unsqueeze(-1)).mean()
        check("cross_entropy_equals_gather", torch.allclose(loss, gathered))
        initial_loss = loss.item()
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % vocab_size
        changed_logits = model(changed)
        causal_error = (logits[:, :8] - changed_logits[:, :8]).abs().max().item()
        check("future_perturbation_preserves_prefix", causal_error < 1e-6)
        check("changed_suffix_can_change_logits", not torch.allclose(
            logits[:, 8:], changed_logits[:, 8:], atol=1e-5))
    try:
        model(torch.zeros(1, context + 1, dtype=torch.long))
    except ValueError:
        check("oversized_context_rejected", True)
    else:
        check("oversized_context_rejected", False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0.01)
    steps = 100 if args.quick else 220
    all_gradients_finite = True
    first_embedding_grad_norm = None
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        all_gradients_finite &= all(p.grad is not None and
            bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if step == 0:
            first_embedding_grad_norm = model.token.weight.grad.norm().item()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        final_loss = F.cross_entropy(final_logits.reshape(-1, vocab_size),
                                     y.reshape(-1)).item()
        accuracy = (final_logits.argmax(-1) == y).float().mean().item()
    check("all_training_gradients_finite", all_gradients_finite)
    check("embedding_gradient_nonzero", first_embedding_grad_norm > 0)
    check("all_final_parameters_finite", all(torch.isfinite(p).all()
                                             for p in model.parameters()))
    check("loss_reduced_by_at_least_75_percent", final_loss < 0.25 * initial_loss)
    check("training_accuracy_at_least_90_percent", accuracy >= 0.9)
    generator = torch.Generator().manual_seed(args.seed)
    generated = model.generate(x[:1, :4], 28, generator)
    check("generated_length", generated.shape == (1, 32))
    check("generation_preserves_prompt", torch.equal(generated[:, :4], x[:1, :4]))
    check("generated_ids_in_vocabulary", generated.min() >= 0 and generated.max()
          < vocab_size)
    report = {"chapter": "03-minimal-gpt", "seed": args.seed, "checks": checks,
              "metrics": {"device": "cpu", "steps": steps,
                  "vocab_size": vocab_size, "batch_size": batch,
                  "sequence_length": length, "width": width, "heads": heads,
                  "layers": layers, "context": context, "parameters": parameters,
                  "untied_parameters": sum(p.numel() for p in untied.parameters()),
                  "initial_loss": initial_loss, "final_loss": final_loss,
                  "training_accuracy": accuracy, "causal_max_abs_error": causal_error,
                  "first_embedding_grad_norm": first_embedding_grad_norm,
                  "generated_text": "".join(chars[i] for i in generated[0].tolist()),
                  "elapsed_seconds": time.perf_counter() - started}}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
