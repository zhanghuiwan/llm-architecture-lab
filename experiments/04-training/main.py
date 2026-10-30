"""Chapter 04: deterministic CPU training, exact resume and token weighting."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


class TinyLM(nn.Module):
    """A deliberately small per-position next-token model, not a full GPT."""
    def __init__(self, vocab: int, dropout: float = 0.25):
        super().__init__()
        self.embed = nn.Embedding(vocab, 16)
        self.hidden = nn.Linear(16, 24)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(24, vocab)

    def forward(self, x):
        return self.head(self.dropout(F.gelu(self.hidden(self.embed(x)))))


def make_data():
    # Split source documents BEFORE making overlapping windows.
    train_docs = ("春风吹过山谷春风吹过山谷", "秋雨落在河边秋雨落在河边")
    validation_docs = ("春雨落在山边春雨落在山边",)
    vocab = sorted(set("春风吹过山谷秋雨落在河边"))  # fixed alphabet, not learned from validation
    stoi = {char: i for i, char in enumerate(vocab)}
    def windows(docs):
        rows = []
        for doc in docs:
            ids = [stoi[c] for c in doc]
            rows.extend(ids[i:i + 6] for i in range(len(ids) - 5))
        return torch.tensor(rows, dtype=torch.long)
    return windows(train_docs), windows(validation_docs), len(vocab), train_docs, validation_docs


def build(vocab: int, steps: int):
    model = TinyLM(vocab).double()
    # Decay matrices, exclude biases and one-dimensional scale parameters.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=0.03, betas=(0.9, 0.95))
    def lr_factor(index):
        if index < 2:
            return (index + 1) / 2
        progress = min(1.0, (index - 2) / max(1, steps - 2))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    return model, optimizer, scheduler


def train_step(model, optimizer, scheduler, data):
    model.train()
    indices = torch.randint(len(data), (4,))  # shares CPU RNG with dropout
    batch = data[indices]
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.75)
    clipped = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()  # step after optimizer, once per optimizer update
    return {
        "loss": float(loss.detach()), "learning_rate": lr,
        "batch_sha256": hashlib.sha256(batch.numpy().tobytes()).hexdigest(),
        "unclipped_norm": float(norm), "clipped_norm": clipped,
    }


def probe_loss(model, data):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(data[:, :-1])
        value = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1)))
    model.train(was_training)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    steps = 12 if args.quick else 30
    split = 5 if args.quick else 11
    data, validation, vocab, train_docs, validation_docs = make_data()
    model, optimizer, scheduler = build(vocab, steps)
    initial_probe = probe_loss(model, data)
    records = []
    checkpoint = None
    for index in range(steps):
        records.append(train_step(model, optimizer, scheduler, data))
        if index + 1 == split:
            checkpoint = copy.deepcopy({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                "next_step": index + 1, "total_steps": steps,
                "data_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
            })
    final_state = copy.deepcopy(model.state_dict())
    final_probe = probe_loss(model, data)
    validation_loss = probe_loss(model, validation)
    assert checkpoint is not None
    # Exercise disk serialization, not just an in-memory state copy.
    with tempfile.TemporaryDirectory(prefix="chapter04-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(checkpoint, path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
    resumed, resumed_opt, resumed_sched = build(vocab, steps)
    resumed.load_state_dict(saved["model"])
    resumed_opt.load_state_dict(saved["optimizer"])
    resumed_sched.load_state_dict(saved["scheduler"])
    # Restore LAST: constructing the new model itself consumed random numbers.
    torch.set_rng_state(saved["cpu_rng"])
    resumed_records = [train_step(resumed, resumed_opt, resumed_sched, data)
                       for _ in range(saved["next_step"], steps)]
    parameter_error = max(float((v - resumed.state_dict()[k]).abs().max())
                          for k, v in final_state.items())
    # Negative control: same checkpoint but a deliberately wrong RNG state.
    wrong, wrong_opt, wrong_sched = build(vocab, steps)
    wrong.load_state_dict(saved["model"])
    wrong_opt.load_state_dict(saved["optimizer"])
    wrong_sched.load_state_dict(saved["scheduler"])
    torch.manual_seed(args.seed + 100003)
    wrong_rng_record = train_step(wrong, wrong_opt, wrong_sched, data)
    # Negative control: reset Adam moments while preserving the correct LR/RNG.
    no_moments, fresh_opt, fresh_sched = build(vocab, steps)
    no_moments.load_state_dict(saved["model"])
    fresh_sched.load_state_dict(saved["scheduler"])
    for group, saved_group in zip(fresh_opt.param_groups, saved["optimizer"]["param_groups"]):
        group["lr"] = saved_group["lr"]
    torch.set_rng_state(saved["cpu_rng"])
    fresh_record = train_step(no_moments, fresh_opt, fresh_sched, data)
    correct_first = copy.deepcopy(saved["model"])
    one_step, one_opt, one_sched = build(vocab, steps)
    one_step.load_state_dict(correct_first)
    one_opt.load_state_dict(saved["optimizer"])
    one_sched.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["cpu_rng"])
    correct_record = train_step(one_step, one_opt, one_sched, data)
    missing_moments_error = max(float((p.detach() - q.detach()).abs().max())
                                for p, q in zip(no_moments.parameters(), one_step.parameters()))
    # Unequal token counts: arithmetic mean of microbatch means is WRONG.
    torch.manual_seed(args.seed + 17)
    base = TinyLM(vocab, dropout=0).double()
    full, accumulated, incorrect = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    x = data[:4, :-1]
    y = data[:4, 1:].clone()
    y[2:, 1:] = -100  # first microbatch has 10 valid tokens, second has 2
    count = int((y != -100).sum())
    def token_sum(m, start, end):
        z = m(x[start:end])
        return F.cross_entropy(z.reshape(-1, vocab), y[start:end].reshape(-1),
                               ignore_index=-100, reduction="sum")
    (token_sum(full, 0, 4) / count).backward()
    for start, end in ((0, 2), (2, 4)):
        (token_sum(accumulated, start, end) / count).backward()
        local_count = int((y[start:end] != -100).sum())
        (token_sum(incorrect, start, end) / (2 * local_count)).backward()
    grad_error = max(float((p.grad - q.grad).abs().max())
                     for p, q in zip(full.parameters(), accumulated.parameters()))
    wrong_grad_error = max(float((p.grad - q.grad).abs().max())
                           for p, q in zip(full.parameters(), incorrect.parameters()))
    checks = {
        "source_documents_split_before_windows": not set(train_docs) & set(validation_docs),
        "training_losses_finite": all(math.isfinite(r["loss"]) for r in records),
        "gradient_clipping_respects_bound": all(r["clipped_norm"] <= 0.750000001 for r in records),
        "schedule_changes_learning_rate": len({r["learning_rate"] for r in records}) > 2,
        "checkpoint_data_identity_matches": saved["data_sha256"] == hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        "resume_records_exactly_equal": resumed_records == records[split:],
        "resume_parameters_bitwise_equal": parameter_error == 0.0,
        "wrong_rng_is_detected": wrong_rng_record["batch_sha256"] != records[split]["batch_sha256"] or wrong_rng_record["loss"] != records[split]["loss"],
        "fresh_optimizer_same_forward_loss": fresh_record["loss"] == correct_record["loss"],
        "missing_adam_moments_changes_update": missing_moments_error > 1e-8,
        "weighted_accumulation_matches_full_gradient": grad_error < 1e-12,
        "equal_microbatch_mean_is_wrong": wrong_grad_error > 1e-7,
        "fixed_probe_training_loss_decreases": final_probe < initial_probe,
    }
    report = {
        "chapter": "04-training", "seed": args.seed, "checks": checks,
        "metrics": {
            "steps": steps, "resume_step": split, "effective_tokens_per_update": 20,
            "training_tokens": steps * 20, "initial_train_probe_loss": initial_probe,
            "final_train_probe_loss": final_probe, "validation_loss": validation_loss,
            "resume_parameter_max_error": parameter_error,
            "weighted_accumulation_gradient_max_error": grad_error,
            "wrong_accumulation_gradient_max_error": wrong_grad_error,
            "missing_optimizer_moments_parameter_max_error": missing_moments_error,
            "trace": records,
        },
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
