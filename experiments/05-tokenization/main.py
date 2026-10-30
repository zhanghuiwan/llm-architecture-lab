#!/usr/bin/env python3
"""A self-contained byte-BPE and document-isolated packing experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random

PAD, BOS, EOS = 256, 257, 258
IGNORE = -100


@dataclass
class ByteBPE:
    merges: list[tuple[int, int, int]]
    pieces: dict[int, bytes]

    @classmethod
    def train(cls, documents: list[str], budget: int) -> "ByteBPE":
        """Pairs are counted within each document; ties use token-ID order."""
        sequences = [list(doc.encode("utf-8")) for doc in documents]
        pieces = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[int, int, int]] = []
        for _ in range(budget):
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts or max(counts.values()) < 2:
                break
            left, right = min(counts, key=lambda pair: (-counts[pair], pair))
            token = 259 + len(merges)
            pieces[token] = pieces[left] + pieces[right]
            sequences = [replace_pair(seq, left, right, token) for seq in sequences]
            merges.append((left, right, token))
        return cls(merges, pieces)

    def encode(self, text: str) -> list[int]:
        # There is deliberately no special-token string parser here.
        result = list(text.encode("utf-8"))
        for left, right, token in self.merges:
            result = replace_pair(result, left, right, token)
        return result

    def decode(self, ids: list[int], *, skip_special: bool = False) -> str:
        chunks: list[bytes] = []
        for token in ids:
            if token in (PAD, BOS, EOS):
                if skip_special:
                    continue
                raise ValueError("Special IDs require explicit skip_special=True")
            if token not in self.pieces:
                raise ValueError(f"Unknown token ID: {token}")
            chunks.append(self.pieces[token])
        # Decode the concatenation, not individual byte pieces.
        return b"".join(chunks).decode("utf-8", errors="strict")


def replace_pair(seq: list[int], left: int, right: int, new: int) -> list[int]:
    result: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
            result.append(new)
            i += 2
        else:
            result.append(seq[i])
            i += 1
    return result


def pack_documents(documents: list[list[int]], block: int) -> list[dict]:
    """Pack without dropping tails, with explicit next-token boundary masking."""
    stream = [
        (token, doc_id, position)
        for doc_id, content in enumerate(documents)
        for position, token in enumerate([BOS, *content, EOS])
    ]
    batches = []
    for start in range(0, len(stream), block):
        chunk = stream[start:start + block]
        ids = [item[0] for item in chunk]
        segment = [item[1] for item in chunk]
        positions = [item[2] for item in chunk]
        labels = []
        for offset, (_, doc_id, _) in enumerate(chunk):
            following = start + offset + 1
            labels.append(stream[following][0]
                          if following < len(stream) and stream[following][1] == doc_id
                          else IGNORE)
        padding = block - len(chunk)
        ids += [PAD] * padding
        labels += [IGNORE] * padding
        segment += [-1] * padding
        positions += [0] * padding
        # True means allowed; this is metadata, not an additive mask.
        attention = [
            [j <= i and segment[i] >= 0 and segment[i] == segment[j]
             for j in range(block)]
            for i in range(block)
        ]
        batches.append(dict(input_ids=ids, labels=labels, segment=segment,
                            position_ids=positions, attention_allowed=attention))
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = [
        "模型读取文本，模型预测下一个词。", "模型读取数据，模型学习结构。",
        "注意力连接上下文，注意力需要因果掩码。", "训练先固定数据，训练再比较模型。",
        "字节组成文本，字节组成序列。", "缓存保存历史，缓存减少重复计算。",
        "模型需要测试，数据需要划分。", "实验记录条件，实验记录结果。",
    ]
    shuffled = corpus.copy()
    random.Random(args.seed).shuffle(shuffled)
    # The split happens BEFORE fitting a tokenizer. The reserved evaluation
    # sentence makes the held-out-only byte pair observable.
    train, validation = shuffled[:6], shuffled[6:] + ["~~ 留出文本 🧪 ~~"]
    budget = 32 if args.quick else 64
    tokenizer = ByteBPE.train(train, budget)
    repeated = ByteBPE.train(train, budget)
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    check("split_is_disjoint", not set(train) & set(validation))
    check("merges_are_deterministic", tokenizer.merges == repeated.merges)
    check("training_order_does_not_break_ties", tokenizer.merges == ByteBPE.train(list(reversed(train)), budget).merges)
    check("held_out_byte_never_enters_merges", all(b"~" not in piece for index, piece in tokenizer.pieces.items() if index >= 259))
    check("merge_ids_do_not_overlap_special_ids", all(token >= 259 for _, _, token in tokenizer.merges))
    check("learned_at_least_one_merge", len(tokenizer.merges) > 0)
    examples = ["", "中文模型", "hello  world\n", "🧪👩🏽‍💻🙂", "e\u0301 与 é", "<|endoftext|>", "\x00\t\r\n", *train, *validation]
    check("unicode_and_whitespace_roundtrip", all(tokenizer.decode(tokenizer.encode(text)) == text for text in examples))
    check("ordinary_text_cannot_inject_special_ids", all(t not in (PAD, BOS, EOS) for t in tokenizer.encode("<|bos|><|eos|><|pad|>")))
    check("explicit_special_skip_is_correct", tokenizer.decode([BOS, *tokenizer.encode("你好"), EOS], skip_special=True) == "你好")
    rejected = False
    try:
        tokenizer.decode([BOS])
    except ValueError:
        rejected = True
    check("special_decode_requires_explicit_policy", rejected)
    check("overlapping_pair_uses_nonoverlapping_replacement", replace_pair([97, 97, 97], 97, 97, 259) == [259, 97])
    # Isolate tie-breaking from whichever natural-language document was held out.
    tie = ByteBPE.train(["abab", "cdcd"], 1)
    check("tie_break_uses_lexicographic_token_ids", tie.merges == [(97, 98, 259)])
    check("known_merge_encodes_in_training_rank_order", tie.encode("ababc") == [259, 259, 99])
    documents = [tokenizer.encode(text) for text in ["你好", "模型", "数据测试🙂"]]
    block = 11
    packed = pack_documents(documents, block)
    padding_fixture = pack_documents([[97]], block)
    check("fixed_block_shapes", all(len(row[key]) == block for row in packed for key in ("input_ids", "labels", "segment", "position_ids", "attention_allowed")))
    check("target_count_matches_each_document_once", sum(t != IGNORE for row in packed for t in row["labels"]) == sum(len(doc) + 1 for doc in documents))
    check("padding_has_no_targets", all(label == IGNORE for row in packed + padding_fixture for token, label in zip(row["input_ids"], row["labels"]) if token == PAD))
    check("eos_does_not_predict_next_document_bos", all(label == IGNORE for row in packed for token, label in zip(row["input_ids"], row["labels"]) if token == EOS))
    check("attention_is_causal_and_document_isolated", all(not allowed or (j <= i and row["segment"][i] >= 0 and row["segment"][i] == row["segment"][j]) for row in packed for i, line in enumerate(row["attention_allowed"]) for j, allowed in enumerate(line)))
    check("valid_token_can_attend_to_itself", all(row["attention_allowed"][i][i] for row in packed for i, token in enumerate(row["input_ids"]) if token != PAD))
    check("padding_attention_rows_are_empty", all(not any(row["attention_allowed"][i]) for row in packed + padding_fixture for i, token in enumerate(row["input_ids"]) if token == PAD))
    check("packing_really_exercises_padding", any(PAD in row["input_ids"] for row in padding_fixture))
    check("packing_really_exercises_document_boundaries", any(len(set(row["segment"]) - {-1}) > 1 for row in packed))
    rebuilt = [[token for row in packed for token, segment in zip(row["input_ids"], row["segment"]) if segment == i] for i in range(len(documents))]
    check("packing_preserves_all_document_tokens", rebuilt == [[BOS, *doc, EOS] for doc in documents])
    expected_targets = [target for doc in documents for target in [*doc, EOS, IGNORE]]
    actual_targets = [label for row in packed for token, label in zip(row["input_ids"], row["labels"]) if token != PAD]
    check("all_shifted_targets_match_independent_reference", actual_targets == expected_targets)
    # Naively shifting the concatenated stream adds N-1 cross-document targets.
    naive_count = sum(len(doc) + 2 for doc in documents) - 1
    check("naive_cross_document_shift_is_detectably_wrong", naive_count - sum(t != IGNORE for t in actual_targets) == len(documents) - 1)
    check("position_ids_reset_at_document_bos", all(position == 0 for row in packed for token, position in zip(row["input_ids"], row["position_ids"]) if token == BOS))
    train_bytes = sum(len(t.encode("utf-8")) for t in train)
    train_tokens = sum(len(tokenizer.encode(t)) for t in train)
    val_bytes = sum(len(t.encode("utf-8")) for t in validation)
    val_tokens = sum(len(tokenizer.encode(t)) for t in validation)
    check("bpe_never_expands_byte_length", train_tokens <= train_bytes and val_tokens <= val_bytes)
    report = {
        "chapter": "05-tokenization", "seed": args.seed, "checks": checks,
        "metrics": {"train_documents": len(train), "validation_documents": len(validation),
                    "merge_budget": budget, "learned_merges": len(tokenizer.merges),
                    "vocab_size_including_special": 259 + len(tokenizer.merges),
                    "train_utf8_bytes": train_bytes, "train_tokens": train_tokens,
                    "train_bytes_per_token": train_bytes / train_tokens,
                    "validation_utf8_bytes": val_bytes, "validation_tokens": val_tokens,
                    "validation_bytes_per_token": val_bytes / val_tokens,
                    "packing_block_size": block, "packed_blocks": len(packed),
                    "packing_valid_targets": sum(t != IGNORE for row in packed for t in row["labels"]),
                    "packing_pad_tokens": sum(t == PAD for row in packed for t in row["input_ids"]),
                    "normalization": "none", "cross_document_attention": False},
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
