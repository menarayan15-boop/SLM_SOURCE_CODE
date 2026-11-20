"""THEIA-SLM architecture extracted from the research notebook.

NOTE:
This is the THEIASLM prototype defined in notebook cell 44. It is NOT
verified as the exact class used by the final 34,510,699-parameter
checkpoint. Do not use it to load that checkpoint without matching its
state_dict keys and configuration first.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rope_params(head_dim, context_length, theta_base=10_000, device="cpu"):
    if head_dim % 2:
        raise ValueError("RoPE head_dim must be even.")
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(context_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    # x: [batch, heads, sequence, head_dim]
    d = x.size(-1)
    x1, x2 = x[..., :d//2], x[..., d//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos[:x.size(-2)].view(1, 1, x.size(-2), d)
    sin = sin[:x.size(-2)].view(1, 1, x.size(-2), d)
    return x * cos + rotated * sin

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups):
        super().__init__()
        assert n_heads % n_kv_groups == 0
        self.n_heads, self.n_kv_groups = n_heads, n_kv_groups
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0 and self.head_dim % 2 == 0
        self.group_size = n_heads // n_kv_groups
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q = self.q_proj(x).view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(b,t,self.n_kv_groups,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(b,t,self.n_kv_groups,self.head_dim).transpose(1,2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,c))

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class THEIABlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups, d_ff):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_groups)
        self.ff = SwiGLU(d_model, d_ff)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x

class THEIASLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_kv_groups=2, n_layers=2, d_ff=384, context=64):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([THEIABlock(d_model,n_heads,n_kv_groups,d_ff) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Small initialization prevents initially saturated tied-embedding logits.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        cos, sin = compute_rope_params(d_model // n_heads, context)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.head.weight = self.embedding.weight
    def forward(self, idx, targets=None):
        _, t = idx.shape
        if t > self.context:
            raise ValueError(f"Sequence length {t} exceeds context {self.context}")
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)), targets.reshape(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20, temperature=1.0, top_k=20):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

print("THEIA-SLM architecture defined.")

