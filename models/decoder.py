"""
MLP Decoder for CMX Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class DecoderHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], num_classes=40, dropout_ratio=0.1, 
                 norm_layer=nn.BatchNorm2d, embed_dim=768, align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        
        c1, c2, c3, c4 = in_channels
        self.linear_c4 = MLP(c4, embed_dim)
        self.linear_c3 = MLP(c3, embed_dim)
        self.linear_c2 = MLP(c2, embed_dim)
        self.linear_c1 = MLP(c1, embed_dim)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim, 1), norm_layer(embed_dim), nn.ReLU(inplace=True))
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
       
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        _c3 = F.interpolate(self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]), size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        _c2 = F.interpolate(self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]), size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        return self.linear_pred(self.dropout(_c) if self.dropout else _c)

