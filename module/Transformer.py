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
        self.embedding_level_2h = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[0], dim * dim),
        )
        self.embedding_level_3h = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[1], dim * dim),
        )
        self.embedding_level_4h = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[2], dim * dim),
        )
        self.embedding_level_5h = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[3], dim * dim),
        )
        self.embedding_level_2f = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[4], dim * dim),
        )
        self.embedding_level_3f = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[5], dim * dim),
        )
        self.embedding_level_4f = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(embedding_level[6], dim * dim),
        )
        # self.recover = nn.Sequential(
        #     nn.Linear(4096, dim)
        # )

        self.transformer = Transformer(dim * dim * input_channel, depth, heads, dim_head, mlp_dim, dropout)

    def initialize(self):
        pass

    def forward(self, feat2h, feat3h, feat4h, feat5h, feat2f, feat3f, feat4f):
        b, c, _, _ = feat2h.shape
        feat2 = self.embedding_level_2h(feat2h)
        feat3 = self.embedding_level_3h(feat3h)
        feat4 = self.embedding_level_4h(feat4h)
        feat5 = self.embedding_level_5h(feat5h)
        feat6 = self.embedding_level_2f(feat2f)
        feat7 = self.embedding_level_3f(feat3f)
        feat8 = self.embedding_level_4f(feat4f)

        feat = torch.cat([feat2, feat3, feat4, feat5, feat6, feat7, feat8], dim=1)
        feat = rearrange(feat, 'b (n c) d -> b n (c d)', c=c)

        feat = self.transformer(feat)
        # feat = self.recover(feat)
        feat = rearrange(feat, 'b n (c d) -> b (n c) d', c=c)
        feat = rearrange(feat, 'b c (h w) -> b c h w', h=32)
        return feat

if __name__ == '__main__':
    input1 = torch.zeros(1, 64, 56, 56)
    input2 = torch.zeros(1, 64, 28, 28)
    input3 = torch.zeros(1, 64, 14, 14)
    input4 = torch.zeros(1, 64, 7, 7)

    input2f = torch.zeros(1, 64, 28, 28)
    input3f = torch.zeros(1, 64, 14, 14)
    input4f = torch.zeros(1, 64, 14, 14)
    mf = MapFuse(32, 64, 4, 4, 256, embedding_level=[56*56, 28*28, 14*14, 7*7, 28*28, 14*14, 14*14])
    output = mf(input1, input2, input3, input4, input2f, input3f, input4f)
    print(output.shape)
