"""
MiT (Mix Transformer) Backbone for RGB-X Semantic Segmentation
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple
from transformers import SegformerModel
from .modules import FRM, FFM


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        return self.dwconv(x.permute(0, 2, 1).reshape(B, C, H, W)).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = self.norm(self.sr(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, -1).permute(0, 2, 1))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        
        # RGB patch embeds
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])
        
        # NIR patch embeds
        self.extra_patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # Stage 1
        self.block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.extra_block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]
        
        # Stage 2
        self.block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.extra_block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur+1], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]
        
        # Stage 3
        self.block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.extra_block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        
        # Stage 4
        self.block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.extra_block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])
        
        # FRM & FFM
        self.FRMs = nn.ModuleList([FRM(dim=embed_dims[i], reduction=1) for i in range(4)])
        self.FFMs = nn.ModuleList([FFM(dim=embed_dims[i], reduction=1, num_heads=num_heads[i], norm_layer=norm_fuse) for i in range(4)])

    def forward_features(self, x_rgb, x_e):
        B = x_rgb.shape[0]
        outs = []
        
        # Stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_e, _, _ = self.extra_patch_embed1(x_e)
        for blk in self.block1: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1: x_e = blk(x_e, H, W)
        x_rgb, x_e = self.norm1(x_rgb), self.extra_norm1(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_rgb, x_e = self.FRMs[0](x_rgb, x_e)
        outs.append(self.FFMs[0](x_rgb, x_e))
        
        # Stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        for blk in self.block2: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2: x_e = blk(x_e, H, W)
        x_rgb, x_e = self.norm2(x_rgb), self.extra_norm2(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_rgb, x_e = self.FRMs[1](x_rgb, x_e)
        outs.append(self.FFMs[1](x_rgb, x_e))
        
        # Stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        for blk in self.block3: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3: x_e = blk(x_e, H, W)
        x_rgb, x_e = self.norm3(x_rgb), self.extra_norm3(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_rgb, x_e = self.FRMs[2](x_rgb, x_e)
        outs.append(self.FFMs[2](x_rgb, x_e))
        
        # Stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for blk in self.block4: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4: x_e = blk(x_e, H, W)
        x_rgb, x_e = self.norm4(x_rgb), self.extra_norm4(x_e)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_rgb, x_e = self.FRMs[3](x_rgb, x_e)
        outs.append(self.FFMs[3](x_rgb, x_e))
        
        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


# MiT Backbone Variants
class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def load_pretrained_from_transformers(model, backbone='mit_b2'):
    """Load pretrained weights from Hugging Face transformers"""
    hf_names = {'mit_b1': 'nvidia/mit-b1', 'mit_b2': 'nvidia/mit-b2', 'mit_b3': 'nvidia/mit-b3', 'mit_b4': 'nvidia/mit-b4'}
    hf_name = hf_names.get(backbone, 'nvidia/mit-b2')
    print(f"Loading pretrained weights from: {hf_name}...")
    
    try:
        hf_model = SegformerModel.from_pretrained(hf_name)
        raw_state_dict = hf_model.state_dict()
        state_dict = {}
        kv_weights, kv_biases = {}, {}
        
        for k, v in raw_state_dict.items():
            new_key = k.replace('encoder.', '')
            for i in range(4):
                new_key = new_key.replace(f'patch_embeddings.{i}', f'patch_embed{i+1}')
                new_key = new_key.replace(f'block.{i}.', f'block{i+1}.')
                new_key = new_key.replace(f'layer_norm.{i}', f'norm{i+1}')
            new_key = new_key.replace('attention.self.query', 'attn.q')
            new_key = new_key.replace('attention.output.dense', 'attn.proj')
            new_key = new_key.replace('attention.self.sr', 'attn.sr')
            new_key = new_key.replace('attention.self.layer_norm', 'attn.norm')
            new_key = new_key.replace('output.dense', 'mlp.fc2')
            new_key = new_key.replace('intermediate.dense', 'mlp.fc1')
            new_key = new_key.replace('dwconv.dwconv', 'mlp.dwconv.dwconv')
            
            if 'attention.self.key' in k:
                base = new_key.replace('attention.self.key', 'attn.kv')
                kv_weights.setdefault(base, {})['key'] = v
                continue
            elif 'attention.self.value' in k:
                base = new_key.replace('attention.self.value', 'attn.kv')
                kv_weights.setdefault(base, {})['value'] = v
                continue
            
            if any(x in new_key for x in ['patch_embed', 'block', 'norm']):
                state_dict[new_key] = v
                if 'patch_embed' in new_key:
                    state_dict[new_key.replace('patch_embed', 'extra_patch_embed')] = v
                elif 'block' in new_key:
                    state_dict[new_key.replace('block', 'extra_block')] = v
                elif 'norm' in new_key and 'attn' not in new_key:
                    state_dict[new_key.replace('norm', 'extra_norm')] = v
        
        for base, kv in kv_weights.items():
            if 'key' in kv and 'value' in kv:
                w = torch.cat([kv['key'], kv['value']], dim=0)
                state_dict[base] = w
                state_dict[base.replace('block', 'extra_block')] = w
        
        model_state = model.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state, strict=False)
        print(f"✓ Loaded {loaded} keys from pretrained!")
        del hf_model
    except Exception as e:
        print(f"Warning: Could not load pretrained: {e}")

