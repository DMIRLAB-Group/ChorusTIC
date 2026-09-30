import torch

from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    """
    Layer Normalization before a layer.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    The MLP (feed forward) block in a transformer.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    The attention block in a transformer.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)




# class Attention(nn.Module):
#     """
#     The attention block in a transformer.
#     """
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         print("Using Gated Attention (G1)...---------------------------------------")
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         # === G1 gate: SDPA输出后门控（head-specific）===
#         self.gate_proj = nn.Linear(dim, inner_dim, bias=True)
#         # 可选：让初始行为更接近 baseline（gate≈1）
#         nn.init.zeros_(self.gate_proj.weight)
#         nn.init.constant_(self.gate_proj.bias, 4.0)  # sigmoid(2)≈0.88，想更接近1可设到4~6

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(
#             t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         # === G1 gate 插入点：在 SDPA 输出 out 后、rearrange 前 ===
#         gate = torch.sigmoid(self.gate_proj(x))  # [b, n, h*d]
#         gate = rearrange(gate, 'b n (h d) -> b h n d', h=self.heads)
#         out = out * gate
#         # print("Gate mean_Attention:", gate.mean().item())
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class Transformer(nn.Module):
    """
    Transformer layer.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


