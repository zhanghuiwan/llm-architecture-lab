#!/usr/bin/env python3
"""Independent CPU checks for a modern pre-norm Transformer block."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def rope(x, positions, base=10000.0):
    """Adjacent even/odd pairing; x [B,H,T,d], positions [T]."""
    d = x.shape[-1]
    if d % 2:
        raise ValueError('RoPE head dimension must be even')
    inv_freq = base ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
    theta = positions.to(dtype=x.dtype, device=x.device)[:, None] * inv_freq[None, :]
    c, s = theta.cos(), theta.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c - odd*s, even*s + odd*c), dim=-1).flatten(-2)


class ModernBlock(nn.Module):
    def __init__(self, d, heads, middle):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.attn_norm = nn.Parameter(torch.ones(d))
        self.ffn_norm = nn.Parameter(torch.ones(d))
        self.q_norm = nn.Parameter(torch.ones(self.dh))
        self.k_norm = nn.Parameter(torch.ones(self.dh))
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, middle, bias=False)
        self.up = nn.Linear(d, middle, bias=False)
        self.down = nn.Linear(middle, d, bias=False)

    def forward(self, x, start=0):
        b, t, d = x.shape
        q, k, v = self.qkv(rms_norm(x, self.attn_norm)).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.dh).transpose(1, 2) for z in (q, k, v)]
        pos = torch.arange(start, start+t, device=x.device)
        q = rope(rms_norm(q, self.q_norm), pos)
        k = rope(rms_norm(k, self.k_norm), pos)
        values = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        x = x + self.out(values.transpose(1, 2).reshape(b, t, d))
        u = rms_norm(x, self.ffn_norm)
        return x + self.down(F.silu(self.gate(u)) * self.up(u))


def run(seed, quick):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    b, t, d, h = (2, 5, 24, 3) if quick else (3, 7, 48, 4)
    dh, middle = d // h, 8*d // 3
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise AssertionError(name)
    def close(a, z, atol=1e-10):
        return torch.allclose(a, z, atol=atol, rtol=1e-9)

    x = torch.randn(b, t, d, dtype=dtype, requires_grad=True)
    gamma = (torch.rand(d, dtype=dtype) + 0.5).requires_grad_()
    normalized = rms_norm(x, gamma)
    reference = torch.stack([torch.stack([row / torch.sqrt(torch.sum(row**2)/d + 1e-6) * gamma for row in batch]) for batch in x])
    check('rms_shape', normalized.shape == x.shape)
    check('rms_loop_reference', close(normalized, reference))
    probe = torch.randn_like(x)
    g1 = torch.autograd.grad((normalized*probe).sum(), (x, gamma), retain_graph=True)
    g2 = torch.autograd.grad((reference*probe).sum(), (x, gamma))
    check('rms_input_and_weight_gradient', all(close(a,z) for a,z in zip(g1,g2)))
    tiny = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    tiny_weight = torch.ones(4, dtype=dtype, requires_grad=True)
    check('rms_finite_difference_gradcheck', torch.autograd.gradcheck(rms_norm, (tiny,tiny_weight)))
    zero = rms_norm(torch.zeros_like(x), gamma)
    check('rms_zero_is_finite', torch.isfinite(zero).all() and torch.equal(zero, torch.zeros_like(zero)))
    check('rms_epsilon_zero_scale_invariance', close(rms_norm(x.detach()*7, gamma, 0), rms_norm(x.detach(),gamma,0)))
    shifted = x.detach() + 3.0
    check('rms_does_not_remove_mean_shift', not close(rms_norm(shifted,gamma),normalized.detach()))
    check('rms_not_layer_norm', not close(normalized, F.layer_norm(x,(d,),gamma,eps=1e-6)))

    q = torch.randn(b,h,t,dh,dtype=dtype,requires_grad=True)
    k = torch.randn_like(q)
    positions = torch.arange(4,4+t)
    rq, rk = rope(q,positions),rope(k,positions)
    check('rope_shape', rq.shape == q.shape)
    check('rope_preserves_pair_norm', close(rq.reshape(b,h,t,dh//2,2).square().sum(-1),q.reshape(b,h,t,dh//2,2).square().sum(-1)))
    check('rope_zero_position_identity', close(rope(q,torch.zeros(t)),q))
    check('rope_inverse_rotation', close(rope(rq,-positions),q))
    check('rope_same_position_dot_product', close((rq*rk).sum(-1),(q*k).sum(-1)))
    scores = rq @ rk.transpose(-2,-1)
    shifted_scores = rope(q,positions+17) @ rope(k,positions+17).transpose(-2,-1)
    check('rope_common_offset_preserves_scores',close(scores,shifted_scores))
    # Applying Q's offset but resetting K's positions changes relative positions.
    bad_scores = rq @ rope(k,torch.arange(t)).transpose(-2,-1)
    check('rope_wrong_key_offset_negative_control',not close(scores,bad_scores))
    frequencies = 10000.0 ** (-torch.arange(0,dh,2,dtype=dtype)/dh)
    check('rope_frequencies_positive_decreasing',bool((frequencies>0).all()) and bool((frequencies[1:]<frequencies[:-1]).all()))
    # A split-half implementation is equivalent only after a matching channel permutation.
    perm = torch.cat((torch.arange(0,dh,2),torch.arange(1,dh,2)))
    split = q[...,perm]
    theta = positions[:,None]*frequencies[None,:]
    a,z = split.chunk(2,-1)
    split_rotated = torch.cat((a*theta.cos()-z*theta.sin(),a*theta.sin()+z*theta.cos()),-1)
    check('rope_pairing_conversion',close(split_rotated,rq[...,perm]))
    check('rope_pairing_without_conversion_is_wrong',not close(split_rotated,rq))
    tiny_rope = torch.randn(1,1,2,4,dtype=dtype,requires_grad=True)
    check('rope_finite_difference_gradcheck',torch.autograd.gradcheck(lambda z:rope(z,torch.arange(2)),(tiny_rope,)))

    gate = torch.randn(d,middle,dtype=dtype,requires_grad=True)/d**0.5
    up = torch.randn(d,middle,dtype=dtype,requires_grad=True)/d**0.5
    down = torch.randn(middle,d,dtype=dtype,requires_grad=True)/middle**0.5
    input_ffn = torch.randn(b,t,d,dtype=dtype,requires_grad=True)
    swiglu = (F.silu(input_ffn@gate)*(input_ffn@up))@down
    combined = input_ffn@torch.cat((gate,up),dim=1)
    gate_fused,up_fused = combined.chunk(2,dim=-1)
    fused = (gate_fused*torch.sigmoid(gate_fused)*up_fused)@down
    check('swiglu_output_shape',swiglu.shape==input_ffn.shape)
    check('swiglu_fused_projection_reference',close(swiglu,fused))
    ffprobe = torch.randn_like(swiglu)
    ff1 = torch.autograd.grad((swiglu*ffprobe).sum(),(input_ffn,gate,up,down),retain_graph=True)
    ff2 = torch.autograd.grad((fused*ffprobe).sum(),(input_ffn,gate,up,down))
    check('swiglu_all_gradient_reference',all(close(a,z) for a,z in zip(ff1,ff2)))
    plain_parameters = 2*d*(4*d)
    gated_parameters = 3*d*middle
    check('swiglu_matched_parameter_budget',gated_parameters==plain_parameters)
    check('same_middle_is_not_same_parameter_budget',3*d*(4*d)>plain_parameters)
    check('swiglu_missing_activation_negative_control',not close(swiglu,((input_ffn@gate)*(input_ffn@up))@down))

    qweight = torch.linspace(.7,1.3,dh,dtype=dtype)
    qn = rms_norm(q,qweight)
    manual_qn = q / (q.square().sum(-1,keepdim=True)/dh+1e-6).sqrt() * qweight
    check('qk_norm_per_head_reference',close(qn,manual_qn))
    wrong_axis = q / (q.square().mean((1,3),keepdim=True)+1e-6).sqrt()*qweight
    check('qk_norm_across_heads_negative_control',not close(qn,wrong_axis))
    check('qk_norm_not_unit_vector',not close(qn.square().sum(-1),torch.ones(b,h,t,dtype=dtype)))

    block = ModernBlock(d,h,middle).to(dtype=dtype)
    bx = torch.randn(b,t,d,dtype=dtype,requires_grad=True)
    by = block(bx)
    check('block_shape',by.shape==bx.shape)
    changed = bx.detach().clone(); changed[:,3:] += torch.randn_like(changed[:,3:])*4
    check('block_causal_prefix',close(block(changed)[:,:3],by[:,:3]))
    check('block_common_rope_offset',close(block(bx,start=11),by))
    by.square().mean().backward()
    check('block_all_gradients_finite',all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()) and torch.isfinite(bx.grad).all())
    expected = 4*d*d+3*d*middle+2*d+2*dh
    check('block_parameter_accounting',sum(p.numel() for p in block.parameters())==expected)
    with torch.no_grad():
        block.out.weight.zero_();block.down.weight.zero_()
    check('pre_norm_zero_branches_are_identity',torch.equal(block(bx.detach()),bx.detach()))

    vocab = 19
    ids = torch.randint(vocab,(b,t))
    targets = torch.randint(vocab,(b,t))
    shared = nn.Parameter(torch.randn(vocab,d,dtype=dtype)/d**.5)
    separate_embed = nn.Parameter(shared.detach().clone())
    separate_head = nn.Parameter(shared.detach().clone())
    logits_tied = F.linear(F.embedding(ids,shared),shared)
    logits_untied = F.linear(F.embedding(ids,separate_embed),separate_head)
    loss_tied = F.cross_entropy(logits_tied.reshape(-1,vocab),targets.reshape(-1))
    loss_untied = F.cross_entropy(logits_untied.reshape(-1,vocab),targets.reshape(-1))
    loss_tied.backward();loss_untied.backward()
    check('tying_same_initial_forward',close(logits_tied,logits_untied))
    check('tying_gradient_is_sum_of_paths',close(shared.grad,separate_embed.grad+separate_head.grad))
    check('tying_both_paths_contribute',separate_embed.grad.norm()>0 and separate_head.grad.norm()>0)
    holder = nn.Module();holder.embedding=nn.Embedding(vocab,d);holder.head=nn.Linear(d,vocab,bias=False)
    holder.head.weight=holder.embedding.weight
    check('tying_parameter_is_same_object',holder.head.weight is holder.embedding.weight)
    check('tying_deduplicates_parameter_count',sum(p.numel() for p in holder.parameters())==vocab*d)
    return {'chapter':'06-modern-block','seed':seed,'checks':checks,'metrics':{
        'quick':quick,'dtype':'float64','shape':[b,t,d],'heads':h,'head_dim':dh,
        'swiglu_middle':middle,'ffn_parameters_matched':gated_parameters,
        'block_parameters':expected,'tied_parameters_saved':vocab*d,
        'rms_forward_max_abs':(normalized-reference).abs().max().item(),
        'rms_gradient_max_abs':max((a-z).abs().max().item() for a,z in zip(g1,g2)),
        'rope_offset_scores_max_abs':(scores-shifted_scores).abs().max().item(),
        'swiglu_gradient_max_abs':max((a-z).abs().max().item() for a,z in zip(ff1,ff2)),
        'tying_gradient_max_abs':(shared.grad-separate_embed.grad-separate_head.grad).abs().max().item()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=json.dumps(run(args.seed,args.quick),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding='utf-8')
    print(output,end='')

if __name__=='__main__':
    main()
