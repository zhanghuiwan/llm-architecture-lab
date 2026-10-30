#!/usr/bin/env python3
"""固定 Token 预算的微型 MHA/GQA 配对实验。仅需标准库与 PyTorch，CPU 可运行。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = '春夏秋冬山川日月风雨天地。'
TRAIN_DOCS = [
    '春风。夏雨。秋月。冬日。' * 4,
    '山川。天地。日月。风雨。' * 4,
    '春山。夏川。秋天。冬地。' * 4,
    '日风。月雨。山天。川地。' * 4,
]
VALID_DOCS = [
    '春日。夏风。秋雨。冬月。' * 3,
    '山月。川日。天雨。地风。' * 3,
]


def digest(tensors):
    h = hashlib.sha256()
    for value in tensors:
        # 本实验 Token ID <256；序列化内容，不使用进程相关的 Python hash。
        h.update(bytes(value.reshape(-1).tolist()))
    return h.hexdigest()


def windows(documents, length):
    stoi = {character: i for i, character in enumerate(ALPHABET)}
    xs, ys = [], []
    for doc in documents:
        ids = torch.tensor([stoi[c] for c in doc], dtype=torch.long)
        for offset in range(len(ids) - length):
            xs.append(ids[offset:offset + length])
            ys.append(ids[offset + 1:offset + length + 1])
    return torch.stack(xs), torch.stack(ys)


class TinyLM(nn.Module):
    def __init__(self, kv_heads, length=12):
        super().__init__()
        self.width, self.heads, self.kv_heads, self.head_dim = 32, 4, kv_heads, 8
        self.embedding = nn.Embedding(len(ALPHABET), self.width)
        self.position = nn.Embedding(length, self.width)
        self.norm1 = nn.LayerNorm(self.width)
        self.q = nn.Linear(self.width, self.heads * self.head_dim, bias=False)
        self.k = nn.Linear(self.width, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(self.width, kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(self.width, self.width, bias=False)
        self.norm2 = nn.LayerNorm(self.width)
        self.up = nn.Linear(self.width, 64, bias=False)
        self.down = nn.Linear(64, self.width, bias=False)
        self.final_norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, len(ALPHABET), bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        self.head.weight = self.embedding.weight

    def forward(self, ids):
        b, t = ids.shape
        x = self.embedding(ids) + self.position(torch.arange(t, device=ids.device))
        z = self.norm1(x)
        q = self.q(z).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(z).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(z).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        group = self.heads // self.kv_heads
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(attention.transpose(1, 2).contiguous().view(b, t, self.width))
        x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return self.head(self.final_norm(x))


def pair_models(seed, length):
    torch.manual_seed(seed)
    mha = TinyLM(4, length)
    gqa = TinyLM(2, length)
    with torch.no_grad():
        reference = dict(mha.named_parameters())
        for name, parameter in gqa.named_parameters():
            source = reference[name]
            if source.shape == parameter.shape:
                parameter.copy_(source)
            else:
                # MHA 头0/2成为GQA的两个KV头；不平均、不暗改尺度。
                selected = source.view(4, 8, 32)[::2]
                parameter.copy_(selected.reshape_as(parameter))
    return mha, gqa


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        return float(F.cross_entropy(model(x).reshape(-1, len(ALPHABET)), y.reshape(-1)))


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def run(seed, quick):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    length, batch_size = 12, 8
    steps, pair_count = (50, 2) if quick else (100, 4)
    xtrain, ytrain = windows(TRAIN_DOCS, length)
    xvalid, yvalid = windows(VALID_DOCS, length)
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise AssertionError(name)

    check('documents_disjoint', not set(TRAIN_DOCS) & set(VALID_DOCS))
    check('alphabet_unique', len(ALPHABET) == len(set(ALPHABET)))
    check('training_labels_shifted', torch.equal(xtrain[:, 1:], ytrain[:, :-1]))
    check('validation_labels_shifted', torch.equal(xvalid[:, 1:], yvalid[:, :-1]))
    check('different_split_payloads', digest([xtrain, ytrain]) != digest([xvalid, yvalid]))
    pairs = []
    for pair_index in range(pair_count):
        pair_seed = seed + pair_index * 101
        generator = torch.Generator().manual_seed(pair_seed + 7001)
        indices = torch.randint(len(xtrain), (steps, batch_size), generator=generator)
        batches = [(xtrain[index], ytrain[index]) for index in indices]
        expected_hash = digest([tensor for batch in batches for tensor in batch])
        mha, gqa = pair_models(pair_seed, length)
        params = {name: dict(model.named_parameters()) for name, model in [('mha', mha), ('gqa', gqa)]}
        common = [name for name, value in params['mha'].items() if value.shape == params['gqa'][name].shape]
        check(f'pair{pair_index}_shared_initial_values', all(torch.equal(params['mha'][n], params['gqa'][n]) for n in common))
        check(f'pair{pair_index}_kv_head_selection', all(torch.equal(params['gqa'][n].view(2, 8, 32), params['mha'][n].view(4, 8, 32)[::2]) for n in ('k.weight', 'v.weight')))
        pair = {'seed': pair_seed, 'planned_batch_sha256': expected_hash,
                'initialization': {'identical_parameter_names': common,
                                   'kv_mapping': 'GQA KV heads copied from MHA heads [0, 2]'},
                'variants': {}}
        for name, model in [('mha', mha), ('gqa', gqa)]:
            prefix = f'pair{pair_index}_{name}'
            probe = xtrain[:2].clone()
            altered = probe.clone()
            altered[:, 6:] = (altered[:, 6:] + 1) % len(ALPHABET)
            with torch.no_grad():
                before, after = model(probe), model(altered)
            check(prefix + '_causal', torch.equal(before[:, :6], after[:, :6]))
            check(prefix + '_output_shape', tuple(before.shape) == (2, length, len(ALPHABET)))
            check(prefix + '_tied_weights', model.embedding.weight is model.head.weight)
            initial_train = evaluate(model, xtrain, ytrain)
            initial_valid = evaluate(model, xvalid, yvalid)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
            observed = []
            consumed_tokens = 0
            finite_gradients = True
            started = time.perf_counter()
            model.train()
            for xb, yb in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb).reshape(-1, len(ALPHABET)), yb.reshape(-1))
                loss.backward()
                finite_gradients = finite_gradients and bool(torch.isfinite(loss)) and all(
                    parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                consumed_tokens += yb.numel()
                observed.extend((xb, yb))
            training_seconds = time.perf_counter() - started
            final_train = evaluate(model, xtrain, ytrain)
            final_valid = evaluate(model, xvalid, yvalid)
            consumed_hash = digest(observed)
            check(prefix + '_finite_gradients', finite_gradients)
            check(prefix + '_finite_parameters', all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            check(prefix + '_training_loss_decreases', final_train < initial_train)
            check(prefix + '_exact_token_budget', consumed_tokens == steps * batch_size * length)
            check(prefix + '_actual_batches_match_plan', consumed_hash == expected_hash)
            cache_bytes = 2 * batch_size * length * model.kv_heads * model.head_dim * 4
            pair['variants'][name] = {
                'parameters': sum(p.numel() for p in model.parameters()),
                'initial_train_loss': initial_train, 'initial_validation_loss': initial_valid,
                'final_train_loss': final_train, 'final_validation_loss': final_valid,
                'training_tokens': consumed_tokens, 'optimizer_steps': steps,
                'actual_batch_sha256': consumed_hash, 'training_seconds': training_seconds,
                'theoretical_fp32_kv_bytes': cache_bytes,
            }
        a, b = pair['variants']['mha'], pair['variants']['gqa']
        check(f'pair{pair_index}_same_actual_batches', a['actual_batch_sha256'] == b['actual_batch_sha256'])
        check(f'pair{pair_index}_same_budget', a['training_tokens'] == b['training_tokens'])
        check(f'pair{pair_index}_gqa_parameter_saving', a['parameters'] - b['parameters'] == 1024)
        check(f'pair{pair_index}_theoretical_cache_halved', a['theoretical_fp32_kv_bytes'] == 2 * b['theoretical_fp32_kv_bytes'])
        pair['validation_loss_delta_gqa_minus_mha'] = b['final_validation_loss'] - a['final_validation_loss']
        pairs.append(pair)
    differences = [p['validation_loss_delta_gqa_minus_mha'] for p in pairs]
    aggregate = {}
    for name in ('mha', 'gqa'):
        values = [p['variants'][name]['final_validation_loss'] for p in pairs]
        aggregate[name] = {'validation_loss_mean': statistics.mean(values),
                           'validation_loss_sample_std': sample_std(values)}
    aggregate['paired_validation_delta'] = {'mean': statistics.mean(differences),
                                           'sample_std': sample_std(differences),
                                           'standard_error': sample_std(differences) / math.sqrt(pair_count)}
    check('pair_count_at_least_two', pair_count >= 2)
    return {'chapter': '13-research', 'seed': seed, 'checks': checks,
            'metrics': {'mode': 'quick' if quick else 'standard', 'device': 'cpu', 'dtype': 'float32',
                        'width': 32, 'query_heads': 4, 'kv_heads': [4, 2], 'layers': 1,
                        'context': length, 'batch_size': batch_size, 'steps_per_variant': steps,
                        'torch_version': torch.__version__,
                        'optimizer': {'name': 'AdamW', 'lr': 0.008, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-8, 'clip_norm': 1.0},
                        'primary_budget': 'equal effective training tokens per variant',
                        'alphabet': ALPHABET, 'train_windows': len(xtrain), 'validation_windows': len(xvalid),
                        'train_sha256': digest([xtrain, ytrain]), 'validation_sha256': digest([xvalid, yvalid]),
                        'pairs': pairs, 'aggregate': aggregate,
                        'unmeasured': ['GPU timing', 'peak memory', 'FLOPs', 'real-language generalization']}}


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
