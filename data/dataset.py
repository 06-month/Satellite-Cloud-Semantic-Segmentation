"""
Dataset for Cloud Segmentation
"""
import random
import numpy as np
import cv2
import torch
from .augmentations import CopyPasteAugmentation, ImageAug, DefaultAug


class CloudDataset(torch.utils.data.Dataset):
    """Cloud Segmentation Dataset with RGB and NIR channels"""
    def __init__(self, rgb_paths, ngr_paths, label_paths, 
                 is_train=True, crop_size=512, use_copy_paste=False):

        self.is_train = is_train
        self.use_copy_paste = use_copy_paste
        self.copy_paste = CopyPasteAugmentation()
        
        # Store paths for filename retrieval
        self.rgb_paths_list = rgb_paths

        # Load images to RAM
        self.rgb_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in rgb_paths]
        self.nir_imgs = [cv2.imread(p)[:, :, 2] for p in ngr_paths]
        
        # Load labels if available (not available for test set)
        if len(label_paths) > 0:
            self.lbl_imgs = [cv2.imread(p) for p in label_paths]
            self.has_label = True
        else:
            self.lbl_imgs = []
            self.has_label = False

        self.train_tf = ImageAug(crop_size)
        self.val_tf = DefaultAug()

    def __len__(self):
        return len(self.rgb_imgs)

    def __getitem__(self, idx):
        # 1. Get images
        rgb = self.rgb_imgs[idx].copy()
        nir = self.nir_imgs[idx].copy()[..., None]  # (H, W) -> (H, W, 1)
        
        # Test case: no labels
        if not self.has_label:
            dummy = np.zeros(rgb.shape[:2], dtype=np.uint8)
            rgb_t, nir_t, _ = self.val_tf(rgb, nir, dummy)
            return (rgb_t, nir_t), self.rgb_paths_list[idx]

        # Train/Val case: with labels
        lbl = self.lbl_imgs[idx].copy()
        
        # Create mask from color labels
        mask = np.zeros(lbl.shape[:2], dtype=np.uint8)
        mask[np.all(lbl == [0,0,255], axis=-1)] = 1  # Thick Cloud (Red)
        mask[np.all(lbl == [0,255,0], axis=-1)] = 2  # Thin Cloud (Green)
        mask[np.all(lbl == [0,255,255], axis=-1)] = 3 # Cloud Shadow (Yellow)

        # Copy-Paste augmentation
        if self.is_train and self.use_copy_paste:
            j = random.randint(0, len(self.rgb_imgs)-1)
            src_rgb = self.rgb_imgs[j]
            src_nir = self.nir_imgs[j][..., None]
            src_lbl = self.lbl_imgs[j]
            src_mask = np.zeros(src_lbl.shape[:2], dtype=np.uint8)
            src_mask[np.all(src_lbl == [0,0,255], axis=-1)] = 1
            src_mask[np.all(src_lbl == [0,255,0], axis=-1)] = 2
            src_mask[np.all(src_lbl == [0,255,255], axis=-1)] = 3
            
            rgb, nir, mask = self.copy_paste(rgb, nir, mask, src_rgb, src_nir, src_mask)

        # Transform
        if self.is_train:
            rgb_t, nir_t, mask_t = self.train_tf(rgb, nir, mask)
        else:
            rgb_t, nir_t, mask_t = self.val_tf(rgb, nir, mask)
        
        return (rgb_t, nir_t), mask_t

