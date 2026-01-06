"""
Data Augmentation for Cloud Segmentation
"""
import random
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CopyPasteAugmentation:
    """Copy-Paste Augmentation for cloud instances"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, nir, mask, src_rgb, src_nir, src_mask):
        if random.random() > self.p:
            return rgb, nir, mask

        H, W = rgb.shape[:2]

        # Binary cloud mask
        cloud = (src_mask > 0).astype(np.uint8)
        if cloud.sum() == 0:
            return rgb, nir, mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cloud, connectivity=8)
        if num_labels <= 1:
            return rgb, nir, mask

        # Pick random component
        comp_idx = random.randint(1, num_labels - 1)
        x, y, w, h, area = stats[comp_idx]
        if area < 100:
            return rgb, nir, mask

        comp = (labels == comp_idx).astype(np.uint8)
        crop_rgb = src_rgb[y:y+h, x:x+w].copy()
        crop_nir = src_nir[y:y+h, x:x+w].copy()
        crop_mask = src_mask[y:y+h, x:x+w].copy()
        crop_comp = comp[y:y+h, x:x+w].copy()

        # Resize with scale
        scale = random.uniform(0.4, 1.2)
        nh, nw = int(h * scale), int(w * scale)
        if nh < 2 or nw < 2:
            return rgb, nir, mask
        
        # Limit size to original image
        nh = min(nh, H)
        nw = min(nw, W)

        crop_rgb = cv2.resize(crop_rgb, (nw, nh))
        crop_nir = cv2.resize(crop_nir, (nw, nh))
        crop_mask = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        crop_comp = cv2.resize(crop_comp, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # Ensure shape (H,W,1) for NIR
        if crop_nir.ndim == 2:
            crop_nir = crop_nir[..., None]

        # Safe position calculation
        px = random.randint(0, max(0, W - nw))
        py = random.randint(0, max(0, H - nh))
        
        # Calculate actual paste area (boundary check)
        paste_h = min(nh, H - py)
        paste_w = min(nw, W - px)
        
        # Crop to same size
        crop_rgb = crop_rgb[:paste_h, :paste_w]
        crop_nir = crop_nir[:paste_h, :paste_w]
        crop_mask = crop_mask[:paste_h, :paste_w]
        crop_comp = crop_comp[:paste_h, :paste_w]
        
        region = crop_comp.astype(bool)

        # Apply paste
        rgb_patch = rgb[py:py+paste_h, px:px+paste_w]
        nir_patch = nir[py:py+paste_h, px:px+paste_w]
        mask_patch = mask[py:py+paste_h, px:px+paste_w]

        rgb_patch[region] = crop_rgb[region]
        nir_patch[region] = crop_nir[region]
        mask_patch[region] = crop_mask[region]

        return rgb, nir, mask


class ImageAug:
    """Training augmentation with geometric and color transforms"""
    def __init__(self, crop_size=512):
        # 1. Geometric transforms (supports 4 channels) - applied to RGB+NIR
        self.geom = A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.2, 
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.5, p=0.5),
            ], p=0.3),
        ])
        
        # 2. Color transforms (3 channels only) - applied to RGB only
        self.color = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ])

        # 3. Normalization (RGB)
        self.normalize_rgb = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        # 4. Normalization (NIR)
        self.normalize_nir = A.Compose([
            A.Normalize(
                mean=[0.5],
                std=[0.25],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

    def __call__(self, rgb, nir, mask):
        if nir.ndim == 2:
            nir = nir[..., None]

        # Step 1: Geometric transforms (RGB+NIR+Mask together)
        combined = np.concatenate([rgb, nir], axis=-1)
        aug = self.geom(image=combined, mask=mask)
        combined = aug["image"]
        mask = aug["mask"]
        
        # Split
        rgb_aug = combined[:, :, :3]
        nir_aug = combined[:, :, 3:]

        # Step 2: Color transforms (RGB only)
        rgb_aug = self.color(image=rgb_aug)["image"]

        # Step 3: Normalize and convert to tensor
        rgb_t = self.normalize_rgb(image=rgb_aug)["image"]
        nir_t = self.normalize_nir(image=nir_aug)["image"]
        mask_t = torch.from_numpy(mask).long()

        return rgb_t, nir_t, mask_t


class DefaultAug:
    """Validation/Test augmentation (normalization only)"""
    def __init__(self):
        self.tf = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5],
                std=[0.229, 0.224, 0.225, 0.25],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

    def __call__(self, rgb, nir, mask):
        if nir.ndim == 2:
            nir = nir[..., None]

        combined = np.concatenate([rgb, nir], axis=-1)

        aug = self.tf(image=combined, mask=mask)
        img4 = aug["image"]
        mask = aug["mask"]

        rgb_t = img4[:3, :, :]
        nir_t = img4[3:4, :, :]
        mask_t = mask.long()

        return rgb_t, nir_t, mask_t

