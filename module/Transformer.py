import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MapFuse(nn.Module):
    def __init__(self, dim, input_channel, depth, heads, mlp_dim, dim_head = 64, embedding_level = [], dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.embedding_level2 = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[0], dim),
        )
        self.embedding_level3 = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[1], dim),
        )
        self.embedding_level4 = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[2], dim),
        )
        # self.recover = nn.Sequential(
        #     nn.Linear(4096, dim)
        # )

        self.transformer = Transformer(dim * input_channel, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, feat_level2, feat_level3, feat_level4):
        b, c, _, _ = feat_level2.shape
        feat2 = self.embedding_level2(feat_level2)
        feat3 = self.embedding_level3(feat_level3)
        feat4 = self.embedding_level4(feat_level4)

        feat = torch.cat([feat2, feat3, feat4], dim=1)
        feat = rearrange(feat, 'b (n c) d -> b n (c d)', c=c)

        feat = self.transformer(feat)
        # feat = self.recover(feat)
        feat = rearrange(feat, 'b n (c d) -> b (n c) d', c=c)
        return feat

if __name__ == '__main__':
    input1 = torch.zeros(1, 32, 56, 56)
    input2 = torch.zeros(1, 32, 28, 28)
    input3 = torch.zeros(1, 32, 14, 14)
    mf = MapFuse(256, 32, 4, 4, 256, embedding_level=[56*56, 28*28, 14*14])
    output = mf(input1, input2, input3)
    print(output.shape)
