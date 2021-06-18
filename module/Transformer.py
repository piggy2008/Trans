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

class Attention2(nn.Module):
    def __init__(self, dim, size):
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
        self.shape = [size, size]
        self.scale = dim ** -0.5
        inner_dim = dim
        self.attend = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.Sigmoid())
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = nn.Sequential(nn.Conv2d(dim, inner_dim * 3, 1, bias=False), nn.BatchNorm2d(inner_dim * 3), nn.ReLU(inplace=True))

        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def initialize(self):
        pass
    def forward(self, feat2h, feat3h, feat4h, feat5h, feat2f, feat3f, feat4f, pred=None):
        feat2 = F.interpolate(self.embedding_level_2h(feat2h), size=self.shape, mode='bilinear')
        feat3 = F.interpolate(self.embedding_level_3h(feat3h), size=self.shape, mode='bilinear')
        feat4 = F.interpolate(self.embedding_level_4h(feat4h), size=self.shape, mode='bilinear')
        feat5 = F.interpolate(self.embedding_level_5h(feat5h), size=self.shape, mode='bilinear')
        feat6 = F.interpolate(self.embedding_level_2f(feat2f), size=self.shape, mode='bilinear')
        feat7 = F.interpolate(self.embedding_level_3f(feat3f), size=self.shape, mode='bilinear')
        feat8 = F.interpolate(self.embedding_level_4f(feat4f), size=self.shape, mode='bilinear')

        if pred is not None:
            pred = F.interpolate(pred, size=self.shape, mode='bilinear')
            feat2 = feat2 + pred
            feat3 = feat3 + pred
            feat4 = feat4 + pred
            feat5 = feat5 + pred
            feat6 = feat6 + pred
            feat7 = feat7 + pred
            feat8 = feat8 + pred


        b, c, h, w = feat2.shape
        qkv1 = self.to_qkv(feat2).chunk(3, dim=1)
        qkv2 = self.to_qkv(feat3).chunk(3, dim=1)
        qkv3 = self.to_qkv(feat4).chunk(3, dim=1)
        qkv4 = self.to_qkv(feat5).chunk(3, dim=1)
        qkv5 = self.to_qkv(feat6).chunk(3, dim=1)
        qkv6 = self.to_qkv(feat7).chunk(3, dim=1)
        qkv7 = self.to_qkv(feat8).chunk(3, dim=1)

        qkv_list = [qkv1, qkv2, qkv3, qkv4, qkv5, qkv6, qkv7]
        q_list, k_list, v_list = [], [], []

        for qkv in qkv_list:
            # q, k, v = map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), qkv)
            q, k, v = qkv
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        output = []
        for i, q in enumerate(q_list):
            result = torch.zeros(b, 64, h, w).cuda()
            for j, k in enumerate(k_list):
                tmp = q + k
                attn = self.attend(tmp)
                # out = einsum('b i j, b j d -> b i d', attn, v_list[i])
                out = attn * v_list[i]
                result = result + out
            output.append(result)

        return self.to_out(output[0]) + feat2, self.to_out(output[1]) + feat3, self.to_out(output[2]) + feat4, \
               self.to_out(output[3]) + feat5, self.to_out(output[4]) + feat6, \
               self.to_out(output[5]) + feat7, self.to_out(output[6]) + feat8

class Attention3(nn.Module):
    def __init__(self, dim):
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

        inner_dim = dim
        self.attend = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid())
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = nn.Sequential(nn.Conv2d(dim, inner_dim * 3, 1, bias=False), nn.BatchNorm2d(inner_dim * 3), nn.ReLU(inplace=True))

        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def initialize(self):
        pass

    def local_attention(self, q_list, k_list, v_list):
        output = []
        for i, q in enumerate(q_list):
            result = torch.zeros_like(v_list[0]).cuda()
            for j, k in enumerate(k_list):
                tmp = q + k
                attn = self.attend(tmp)
                # out = einsum('b i j, b j d -> b i d', attn, v_list[i])
                out = attn * v_list[i]
                result = result + out
            output.append(result)
        return output

    def forward(self, feat2h, feat3h, feat4h, feat5h, feat2f, feat3f, feat4f, pred=None):

        feat4 = self.embedding_level_4h(feat4h)
        feat5 = F.interpolate(self.embedding_level_5h(feat5h), size=feat4.shape[2:], mode='bilinear')
        feat8 = F.interpolate(self.embedding_level_4f(feat4f), size=feat4.shape[2:], mode='bilinear')
        if pred is not None:
            pred = F.interpolate(pred, size=feat4.shape[2:], mode='bilinear')
            feat4 = feat4 + pred
            feat5 = feat5 + pred
            feat8 = feat8 + pred
        q4, k4, v4 = self.to_qkv(feat4).chunk(3, dim=1)
        q5, k5, v5 = self.to_qkv(feat5).chunk(3, dim=1)
        q8, k8, v8 = self.to_qkv(feat8).chunk(3, dim=1)

        q_list, k_list, v_list = [q4, q5, q8], [k4, k5, k5], [v4, v5, v8]
        output = self.local_attention(q_list, k_list, v_list)
        feat4_refine = self.to_out(output[0]) + feat4
        feat5_refine = self.to_out(output[1]) + feat5
        feat8_refine = self.to_out(output[2]) + feat8

        feat3 = self.embedding_level_3h(feat3h)
        feat7 = F.interpolate(self.embedding_level_3f(feat3f), size=feat3.shape[2:], mode='bilinear')
        feat4_refine = F.interpolate(feat4_refine, size=feat3.shape[2:], mode='bilinear')
        q3, k3, v3 = self.to_qkv(feat3).chunk(3, dim=1)
        q7, k7, v7 = self.to_qkv(feat7).chunk(3, dim=1)
        q4, k4, v4 = self.to_qkv(feat4_refine).chunk(3, dim=1)
        q_list, k_list, v_list = [q3, q7, q4], [k3, k7, k4], [v3, v7, v4]
        output = self.local_attention(q_list, k_list, v_list)
        feat3_refine = self.to_out(output[0]) + feat3
        feat7_refine = self.to_out(output[1]) + feat7

        feat2 = self.embedding_level_2h(feat2h)
        feat6 = F.interpolate(self.embedding_level_2f(feat2f), size=feat2.shape[2:], mode='bilinear')
        feat3_refine = F.interpolate(feat3_refine, size=feat2.shape[2:], mode='bilinear')
        q2, k2, v2 = self.to_qkv(feat2).chunk(3, dim=1)
        q6, k6, v6 = self.to_qkv(feat6).chunk(3, dim=1)
        q3, k3, v3 = self.to_qkv(feat3_refine).chunk(3, dim=1)
        q_list, k_list, v_list = [q2, q6, q3], [k2, k6, k3], [v2, v6, v3]
        output = self.local_attention(q_list, k_list, v_list)
        feat2_refine = self.to_out(output[0]) + feat2
        feat6_refine = self.to_out(output[1]) + feat6


        return feat2_refine, feat3_refine, feat4_refine, feat5_refine, feat6_refine, feat7_refine, feat8_refine

class Transformer(nn.Module):
    def __init__(self, dim, size, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm([dim, size, size], Attention2(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
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
    # mf = MapFuse(64, 56, 4, 4, 256)
    att = Attention3(64)
    # output = mf(input1, input2, input3, input4, input2f, input3f, input4f)
    output1, output2, output3, output4, output5, output6, output7 = att(input1, input2, input3, input4, input2f, input3f, input4f)
    print(output1.shape)
