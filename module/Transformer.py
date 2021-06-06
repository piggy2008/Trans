import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

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
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=inner_dim * 3, kernel_size=1, bias = False)

        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w, head = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (head c) h w -> b head c (h w)', head = head), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=head, h=h, w=h)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, size, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm([dim, size, size], Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm([dim, size, size], FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MapFuse(nn.Module):
    def __init__(self, input_channel, size, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.embedding_level_2h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.embedding_level_3h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.embedding_level_4h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.embedding_level_5h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.embedding_level_2f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.embedding_level_3f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.embedding_level_4f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.transformer = Transformer(input_channel * 7, size, depth, heads, dim_head, mlp_dim, dropout)
        self.shape = [size, size]
    def initialize(self):
        pass

    def forward(self, feat2h, feat3h, feat4h, feat5h, feat2f, feat3f, feat4f):
        b, c, _, _ = feat2h.shape
        feat2 = F.interpolate(self.embedding_level_2h(feat2h), size=self.shape, mode='bilinear')
        feat3 = F.interpolate(self.embedding_level_3h(feat3h), size=self.shape, mode='bilinear')
        feat4 = F.interpolate(self.embedding_level_4h(feat4h), size=self.shape, mode='bilinear')
        feat5 = F.interpolate(self.embedding_level_5h(feat5h), size=self.shape, mode='bilinear')
        feat6 = F.interpolate(self.embedding_level_2f(feat2f), size=self.shape, mode='bilinear')
        feat7 = F.interpolate(self.embedding_level_3f(feat3f), size=self.shape, mode='bilinear')
        feat8 = F.interpolate(self.embedding_level_4f(feat4f), size=self.shape, mode='bilinear')

        feat = torch.cat([feat2, feat3, feat4, feat5, feat6, feat7, feat8], dim=1)
        # feat = rearrange(feat, 'b (n c) d -> b n (c d)', c=c)

        feat = self.transformer(feat)
        # feat = self.recover(feat)
        # feat = rearrange(feat, 'b n (c d) -> b (n c) d', c=c)
        # feat = rearrange(feat, 'b c (h w) -> b c h w', h=32)
        return feat

if __name__ == '__main__':
    input1 = torch.zeros(1, 64, 56, 56)
    input2 = torch.zeros(1, 64, 28, 28)
    input3 = torch.zeros(1, 64, 14, 14)
    input4 = torch.zeros(1, 64, 7, 7)

    input2f = torch.zeros(1, 64, 28, 28)
    input3f = torch.zeros(1, 64, 14, 14)
    input4f = torch.zeros(1, 64, 14, 14)
    mf = MapFuse(64, 56, 4, 4, 256)
    output = mf(input1, input2, input3, input4, input2f, input3f, input4f)
    print(output.shape)
