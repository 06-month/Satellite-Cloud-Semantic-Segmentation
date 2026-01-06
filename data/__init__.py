"""
Data package for Cloud Segmentation
"""
from .dataset import CloudDataset
from .augmentations import ImageAug, DefaultAug, CopyPasteAugmentation

__all__ = ['CloudDataset', 'ImageAug', 'DefaultAug', 'CopyPasteAugmentation']

