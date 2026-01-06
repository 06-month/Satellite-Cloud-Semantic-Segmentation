"""
CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation
"""
import torch.nn as nn
from .backbone import mit_b1, mit_b2, mit_b3, mit_b4, load_pretrained_from_transformers
from .decoder import DecoderHead


class CMXModel(nn.Module):
    """CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation"""
    def __init__(self, backbone='mit_b2', num_classes=4, decoder_embed_dim=512, 
                 pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        backbones = {'mit_b1': mit_b1, 'mit_b2': mit_b2, 'mit_b3': mit_b3, 'mit_b4': mit_b4}
        self.backbone = backbones.get(backbone, mit_b2)(norm_fuse=norm_layer)
        channels = [64, 128, 320, 512]  # mit_b1, mit_b2, mit_b3, mit_b4 share same channel dimensions
        self.decode_head = DecoderHead(in_channels=channels, num_classes=num_classes, 
                                        norm_layer=norm_layer, embed_dim=decoder_embed_dim)
    
    def forward(self, rgb, nir):
        if nir.shape[1] == 1:
            nir = nir.repeat(1, 3, 1, 1)
        features = self.backbone(rgb, nir)
        return self.decode_head(features)


def create_model(device, num_classes=4, cmx_backbone='mit_b2', cmx_pretrained=None, use_timm_pretrained=True):
    """Create CMX model with optional pretrained weights"""
    model = CMXModel(backbone=cmx_backbone, num_classes=num_classes)
    if use_timm_pretrained and cmx_pretrained is None:
        load_pretrained_from_transformers(model.backbone, cmx_backbone)
    model.to(device)
    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

