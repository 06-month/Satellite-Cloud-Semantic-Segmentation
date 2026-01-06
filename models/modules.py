"""
Feature Rectify Module (FRM) and Feature Fusion Module (FFM)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Feature Rectify Module (FRM) ==========
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max_val = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max_val), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        return y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super().__init__()
        self.lambda_c, self.lambda_s = lambda_c, lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def forward(self, x1, x2):
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2
        out_x2 = x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1
        return out_x1, out_x2


FRM = FeatureRectifyModule


# ========== Feature Fusion Module (FFM) ==========
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads = dim, num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        x1 = (q1 @ ctx2.softmax(dim=-2)).permute(0, 2, 1, 3).reshape(B, N, C)
        x2 = (q2 @ ctx1.softmax(dim=-2)).permute(0, 2, 1, 3).reshape(B, N, C)
        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1, self.act2 = nn.ReLU(inplace=True), nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        return self.norm1(x1 + self.end_proj1(torch.cat((y1, v1), dim=-1))), self.norm2(x2 + self.end_proj2(torch.cat((y2, v2), dim=-1)))


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//reduction, 1, bias=True),
            nn.Conv2d(out_channels//reduction, out_channels//reduction, 3, 1, 1, bias=True, groups=out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//reduction, out_channels, 1, bias=True),
            norm_layer(out_channels))
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W)
        return self.norm(self.residual(x) + self.channel_embed(x))


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(dim*2, dim, reduction, norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1, x2 = x1.flatten(2).transpose(1, 2), x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        return self.channel_emb(torch.cat((x1, x2), dim=-1), H, W)


FFM = FeatureFusionModule

