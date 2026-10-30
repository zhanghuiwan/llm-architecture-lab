#!/usr/bin/env python3
"""Self-contained CPU Top-k MoE: dispatch, gradients, capacity, and training."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(torch.tanh(self.up(x)))


class TinyMoE(nn.Module):
    def __init__(self, width=6, hidden=12, experts=4, top_k=2):
        super().__init__()
        self.router = nn.Linear(width, experts, bias=True)
        self.experts = nn.ModuleList(Expert(width, hidden) for _ in range(experts))
        self.shared = Expert(width, hidden)
        self.top_k = top_k

    def route(self, x):
        logits = self.router(x)
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = values.softmax(-1)
        return logits, indices, weights

    def forward(self, x, capacity=None, reference=False):
        logits, indices, weights = self.route(x)
        n, width = x.shape
        counts = torch.bincount(indices.flatten(), minlength=len(self.experts))
        accepted = torch.zeros_like(counts)
        dropped = torch.zeros(n, self.top_k, dtype=torch.bool, device=x.device)
        if reference:
            # Independent per-token oracle. Preserve original weights after drops.
            outputs = []
            for token in range(n):
                result = self.shared(x[token:token+1]).squeeze(0)
                for slot in range(self.top_k):
                    expert_id = int(indices[token, slot])
                    if capacity is None or int(accepted[expert_id]) < capacity:
                        result = result + weights[token, slot] * self.experts[expert_id](x[token:token+1]).squeeze(0)
                        accepted[expert_id] += 1
                    else:
                        dropped[token, slot] = True
                outputs.append(result)
            output = torch.stack(outputs)
        else:
            routed = torch.zeros_like(x)
            # Group assigned tokens by expert, execute, then scatter-add.
            for expert_id, expert in enumerate(self.experts):
                token_ids, slots = (indices == expert_id).nonzero(as_tuple=True)
                keep = len(token_ids) if capacity is None else min(capacity, len(token_ids))
                dropped[token_ids[keep:], slots[keep:]] = True
                selected, selected_slots = token_ids[:keep], slots[:keep]
                accepted[expert_id] = keep
                if keep:
                    contribution = expert(x.index_select(0, selected)) * weights[selected, selected_slots, None]
                    routed = routed.index_add(0, selected, contribution)
            output = self.shared(x) + routed
        probabilities = logits.softmax(-1)
        # Hard assignment frequencies include pre-capacity assignments and sum to 1.
        frequencies = counts.to(x.dtype) / (n * self.top_k)
        balance = len(self.experts) * (frequencies.detach() * probabilities.mean(0)).sum()
        return output, {'logits': logits, 'indices': indices, 'weights': weights,
                        'counts': counts, 'accepted': accepted, 'dropped': dropped,
                        'balance': balance}


def grad_or_zero(parameter):
    return torch.zeros_like(parameter) if parameter.grad is None else parameter.grad


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    n, width, hidden, expert_count, top_k = 16, 6, 12, 4, 2
    model = TinyMoE(width, hidden, expert_count, top_k)
    x = torch.randn(n, width)
    # Spread assignments while retaining distinct, smooth selected scores.
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.weight[:, :expert_count] = 2 * torch.eye(expert_count)
        model.router.bias.copy_(torch.linspace(-.09, .09, expert_count))
    for i in range(expert_count):
        x[i] = .1 * torch.randn(width)
        x[i, i] = 3
    output, state = model(x)
    check('output_shape', output.shape == x.shape)
    check('selected_weights_sum_to_one', torch.allclose(state['weights'].sum(-1), torch.ones(n)))
    check('selected_indices_are_unique', bool((state['indices'][:, 0] != state['indices'][:, 1]).all()))
    full = state['logits'].softmax(-1).gather(1, state['indices'])
    check('selected_softmax_equals_renormalized_global', torch.allclose(state['weights'], full/full.sum(-1, keepdim=True)))
    check('all_experts_receive_tokens', bool((state['counts'] > 0).all()))
    check('dropless_assignment_count', int(state['accepted'].sum()) == n * top_k)
    check('dropless_has_no_dropped_assignments', not bool(state['dropped'].any()))
    sorted_logits = state['logits'].sort(-1, descending=True).values
    boundary_margin = float((sorted_logits[:, top_k-1] - sorted_logits[:, top_k]).min().detach())
    check('routing_boundary_not_tied', boundary_margin > 1e-10)

    comparisons = []
    for label, capacity in [('dropless', None), ('zero', 0), ('one', 1)]:
        fast, slow = copy.deepcopy(model), copy.deepcopy(model)
        xf = x.clone().requires_grad_()
        xs = x.clone().requires_grad_()
        yf, sf = fast(xf, capacity)
        ys, ss = slow(xs, capacity, reference=True)
        error = float((yf - ys).abs().max().detach())
        check(label+'_forward_reference', torch.allclose(yf, ys, atol=1e-11, rtol=1e-10))
        check(label+'_accepted_reference', torch.equal(sf['accepted'], ss['accepted']))
        check(label+'_drop_mask_reference', torch.equal(sf['dropped'], ss['dropped']))
        probe = torch.linspace(-.7, 1.1, n*width).reshape(n, width)
        (yf.mul(probe).sum() + .03 * sf['balance']).backward()
        (ys.mul(probe).sum() + .03 * ss['balance']).backward()
        gradient_error = float((xf.grad - xs.grad).abs().max())
        check(label+'_input_gradient_reference', torch.allclose(xf.grad, xs.grad, atol=1e-11, rtol=1e-10))
        parameter_error = 0.
        for (name, pf), (name_s, ps) in zip(fast.named_parameters(), slow.named_parameters()):
            assert name == name_s
            gf, gs = grad_or_zero(pf), grad_or_zero(ps)
            parameter_error = max(parameter_error, float((gf-gs).abs().max()))
            check(label+'_gradient_'+name, torch.allclose(gf, gs, atol=1e-11, rtol=1e-10))
        check(label+'_assignment_accounting', int(sf['accepted'].sum()+sf['dropped'].sum()) == n*top_k)
        if capacity == 0:
            check('zero_capacity_preserves_shared_path', torch.equal(yf, fast.shared(xf)))
            check('zero_capacity_drops_every_routed_assignment', int(sf['dropped'].sum()) == n*top_k)
        if capacity == 1:
            check('one_capacity_limits_each_expert', bool((sf['accepted'] <= 1).all()))
            check('one_capacity_accepts_four_assignments', int(sf['accepted'].sum()) == expert_count)
        comparisons.append({'capacity': capacity, 'forward_max_error': error,
                            'input_gradient_max_error': gradient_error,
                            'parameter_gradient_max_error': parameter_error,
                            'accepted': sf['accepted'].tolist(), 'dropped_assignments': int(sf['dropped'].sum())})

    # Show collapse explicitly; no claim that this diagnostic solves it.
    collapsed = copy.deepcopy(model)
    with torch.no_grad():
        collapsed.router.weight.zero_()
        collapsed.router.bias.copy_(torch.tensor([12., 8., -8., -12.]))
    _, collapse = collapsed(x)
    check('collapsed_router_leaves_half_experts_idle', int((collapse['counts'] == 0).sum()) == 2)
    uniform_balance = expert_count * torch.dot(torch.full((expert_count,), 1/expert_count), torch.full((expert_count,), 1/expert_count))
    check('uniform_balance_baseline_one', abs(float(uniform_balance)-1) < 1e-12)
    check('collapse_exceeds_uniform_balance', float(collapse['balance'].detach()) > 1.9)
    expert_parameters = 2*width*hidden
    expected_parameters = width*expert_count+expert_count+(expert_count+1)*expert_parameters
    check('parameter_count_matches_formula', sum(p.numel() for p in model.parameters()) == expected_parameters)
    active_parameters = width*expert_count+expert_count+(top_k+1)*expert_parameters
    check('active_parameter_proxy_below_total', active_parameters < expected_parameters)

    # Real fixed-data optimization; deliberately no validation or language claim.
    train_model = TinyMoE(width, hidden, expert_count, top_k)
    tx = torch.randn(40 if quick else 80, width)
    linear_target = torch.randn(width, width) / width**.5
    nonlinear_target = torch.randn(width, width) / width**.5
    ty = .6 * (tx @ linear_target) + .3 * torch.tanh(tx @ nonlinear_target)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=.025, weight_decay=0.)
    with torch.no_grad():
        initial = float(F.mse_loss(train_model(tx)[0], ty))
    steps = 80 if quick else 160
    losses = []
    finite_gradients = True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, info = train_model(tx)
        mse = F.mse_loss(prediction, ty)
        objective = mse + .01 * info['balance']
        objective.backward()
        finite_gradients &= all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in train_model.parameters())
        optimizer.step()
        losses.append(float(mse.detach()))
    with torch.no_grad():
        prediction, trained = train_model(tx)
        final = float(F.mse_loss(prediction, ty))
    check('training_gradients_finite', finite_gradients)
    check('training_loss_decreases', final < initial*.5)
    check('training_predictions_finite', bool(torch.isfinite(prediction).all()))
    return {'chapter': '11-moe', 'seed': seed, 'checks': checks, 'metrics': {
        'mode': 'quick' if quick else 'standard', 'dtype': 'float64', 'threads': 1,
        'tokens': n, 'width': width, 'expert_hidden': hidden, 'experts': expert_count,
        'top_k': top_k, 'route_counts': state['counts'].tolist(), 'routing_boundary_margin': boundary_margin,
        'capacity_comparisons': comparisons,
        'collapse_counts': collapse['counts'].tolist(), 'collapse_balance': float(collapse['balance'].detach()),
        'total_parameters': expected_parameters, 'active_parameter_proxy_per_token': active_parameters,
        'training': {'samples': len(tx), 'steps': steps, 'initial_mse': initial, 'final_mse': final,
                     'final_pre_capacity_counts': trained['counts'].tolist(), 'balance_coefficient': .01,
                     'first_step_mse': losses[0], 'last_pre_update_mse': losses[-1]},
        'elapsed_seconds': time.perf_counter()-started}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.quick)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')

if __name__ == '__main__':
    main()
