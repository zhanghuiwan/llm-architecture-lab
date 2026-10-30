#!/usr/bin/env python3
"""Independent CPU checks for local/global attention and long-context claims."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def make_mask(query_positions, key_positions, window=None):
    distance = query_positions[:, None] - key_positions[None, :]
    allowed = distance >= 0
    return allowed if window is None else allowed & (distance < window)


def manual_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    return scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v


def reference_rows(q, k, v, mask):
    rows = []
    for i in range(q.shape[-2]):
        keys = torch.nonzero(mask[i], as_tuple=True)[0]
        scores = (q[..., i:i+1, :] @ k[..., keys, :].transpose(-2, -1)) / q.shape[-1] ** 0.5
        rows.append(scores.softmax(-1) @ v[..., keys, :])
    return torch.cat(rows, dim=-2)


def reachable(masks):
    # Rows: output positions; columns: original input positions.
    result = torch.eye(masks[0].shape[0], dtype=torch.int64)
    for mask in masks:
        result = ((mask.to(torch.int64) @ result) > 0).to(torch.int64)
    return result.bool()


def controlled_network(x, masks):
    # Q=K=0 gives uniform weights over allowed positions. Residual preserves self.
    for mask in masks:
        weights = mask.to(x.dtype) / mask.sum(-1, keepdim=True)
        x = x + weights @ x
    return x


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    t, window, layers, dim = (12, 3, 2, 5) if quick else (20, 4, 3, 7)
    positions = torch.arange(t)
    global_mask = make_mask(positions, positions)
    local_mask = make_mask(positions, positions, window)
    local_stack = [local_mask] * layers
    mixed_stack = [local_mask] * (layers - 1) + [global_mask]
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    check('global_is_lower_triangular', torch.equal(global_mask, torch.ones(t,t,dtype=torch.bool).tril()))
    check('local_includes_self', local_mask.diag().all())
    check('local_never_reads_future', not local_mask.triu(1).any())
    expected_rows = torch.minimum(positions + 1, torch.tensor(window))
    check('window_includes_self_and_has_expected_size', torch.equal(local_mask.sum(-1), expected_rows))
    q, k, v = [torch.randn(2, 2, t, dim, dtype=dtype, requires_grad=True) for _ in range(3)]
    numerical_errors = {}
    gradient_errors = {}
    for name, mask in [('global', global_mask), ('local', local_mask)]:
        manual = manual_attention(q,k,v,mask)
        rows = reference_rows(q,k,v,mask)
        sdpa = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        error = float((manual - sdpa).abs().max().detach())
        numerical_errors[name] = error
        check(name+'_row_reference_equal', torch.allclose(manual, rows, atol=1e-11, rtol=1e-10))
        check(name+'_sdpa_equal', error < 1e-10)
        probe = torch.randn_like(manual)
        ga = torch.autograd.grad((manual*probe).sum(),(q,k,v),retain_graph=True)
        gb = torch.autograd.grad((sdpa*probe).sum(),(q,k,v),retain_graph=True)
        gradient_errors[name] = max(float((a-b).abs().max()) for a,b in zip(ga,gb))
        check(name+'_all_input_gradients_equal', gradient_errors[name] < 1e-9)
    local_reach = reachable(local_stack)
    expected_reach = make_mask(positions,positions,layers*(window-1)+1)
    check('stacked_local_exact_receptive_field', torch.equal(local_reach,expected_reach))
    check('mixed_last_global_reaches_all_past', torch.equal(reachable(mixed_stack),global_mask))
    check('local_final_query_cannot_reach_first_token', not local_reach[-1,0])
    boundary = layers*(window-1)
    x = torch.randn(t,dim,dtype=dtype)
    changed=x.clone(); changed[0,0]+=1
    local_delta=(controlled_network(changed,local_stack)-controlled_network(x,local_stack)).abs().amax(-1)
    mixed_delta=(controlled_network(changed,mixed_stack)-controlled_network(x,mixed_stack)).abs().amax(-1)
    check('unreachable_input_has_exactly_zero_effect', local_delta[-1] == 0)
    check('reachable_boundary_has_positive_effect', local_delta[boundary] > 1e-8)
    check('global_layer_enables_distant_effect', mixed_delta[-1] > 1e-8)
    future_changed=x.clone(); future_changed[-1,0]+=10
    check('mixed_stack_remains_causal', torch.equal(controlled_network(future_changed,mixed_stack)[:-1],controlled_network(x,mixed_stack)[:-1]))

    # Two queries in a decode chunk need different windows. Cropping to final W keys early is wrong.
    chunk=2
    query_positions=positions[-chunk:]
    full_output=manual_attention(q.detach(),k.detach(),v.detach(),local_mask)[...,-chunk:,:]
    keep_start=max(0,int(query_positions[0])-window+1)
    mask_correct=make_mask(query_positions,positions[keep_start:],window)
    correct=manual_attention(q.detach()[...,-chunk:,:],k.detach()[...,keep_start:,:],v.detach()[...,keep_start:,:],mask_correct)
    early_start=t-window
    wrong=manual_attention(q.detach()[...,-chunk:,:],k.detach()[...,early_start:,:],v.detach()[...,early_start:,:],make_mask(query_positions,positions[early_start:],window))
    cache_error=float((correct-full_output).abs().max())
    early_error=float((wrong-full_output).abs().max())
    check('per_query_chunk_window_matches_full',cache_error < 1e-10)
    check('early_crop_negative_control_detected',early_error > 1e-8)
    check('last_query_can_drop_to_W_keys_after_chunk',torch.allclose(wrong[...,-1,:],full_output[...,-1,:],atol=1e-11,rtol=1e-10))

    # Constructed content-addressing oracle, not a trained retrieval model.
    # One unique orthogonal key per record. Final query is the first key.
    keys=torch.eye(t,dtype=dtype)
    query=keys[0:1]*20
    values=torch.arange(t,dtype=dtype).unsqueeze(-1)
    values[0]=100
    final_global=make_mask(torch.tensor([t-1]),positions)
    final_local=make_mask(torch.tensor([t-1]),positions,window)
    # Here score scale is explicitly 1, so the matching record receives logit20.
    scores=query@keys.T
    def retrieve(mask):
        weights=scores.masked_fill(~mask,-torch.inf).softmax(-1)
        return float((weights@values).item()),float(weights[0,0])
    global_value,global_target_weight=retrieve(final_global)
    local_value,local_target_weight=retrieve(final_local)
    check('oracle_global_retrieves_constructed_target',abs(global_value-100)<1e-4)
    check('oracle_local_target_weight_zero',local_target_weight == 0)
    check('oracle_local_cannot_retrieve_unseen_value',abs(local_value-100)>1)
    return {'chapter':'10-long-context','seed':seed,'checks':checks,'metrics':{
        'mode':'quick' if quick else 'standard','dtype':str(dtype),'sequence_length':t,'window_including_self':window,'layers':layers,
        'local_receptive_field_tokens':layers*(window-1)+1,
        'global_edges_per_layer':int(global_mask.sum()),'local_edges_per_layer':int(local_mask.sum()),
        'mixed_layer_order':['local']*(layers-1)+['global'],
        'forward_max_abs_errors':numerical_errors,'gradient_max_abs_errors':gradient_errors,
        'distant_local_effect':float(local_delta[-1]),'reachable_boundary_effect':float(local_delta[boundary]),'distant_mixed_effect':float(mixed_delta[-1]),
        'correct_chunk_max_abs_error':cache_error,'early_crop_max_abs_error':early_error,
        'oracle':{'target':100,'global_output':global_value,'local_output':local_value,'global_target_weight':global_target_weight,'local_target_weight':local_target_weight,'trained':False},
        'limitations':['CPU FP64 mathematical checks','constructed oracle is not a trained model','no RoPE extrapolation or real-model reproduction'],
    }}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=run(args.seed,args.quick)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding='utf-8')
    print(payload,end='')

if __name__=='__main__':
    main()
