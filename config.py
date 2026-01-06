"""
Configuration file for Cloud Segmentation - CMX Model
"""
import torch

# ========== Paths ==========
workspace_path = '/kaggle/input/clouds-segmentation-2025'
output_path = '/kaggle/working'

# ========== Training Hyperparameters ==========
batch_size = 4
epochs = 60
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4
num_classes = 4

# ========== Data Settings ==========
patch_size = 512
patch_stride = 512 // 4
train_data_rate = 0.8

# ========== Model Settings ==========
cmx_backbone = 'mit_b2'  # Options: 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'
cmx_pretrained = True
use_timm_pretrained = True

# ========== Loss Settings ==========
loss_func = 'ohem+dice'  # Options: 'dice', 'ce', 'jaccard', 'dice+ce', 'ohem+dice'

# ========== Optimizer Settings ==========
lr_head = 3e-4
lr_backbone = 3e-5
weight_decay = 2e-2
accumulation_steps = 4

# ========== Scheduler Settings ==========
use_scheduler = True
scheduler_type = "cosine"  # Options: "cosine", "plateau"

# ========== Augmentation Settings ==========
use_copy_paste = True

# ========== Training Settings ==========
resume = False
seed = 0
val_interval = 5

print(f"Device: {device}, Batch: {batch_size}, Epochs: {epochs}")
print(f"Augmentation: Copy-Paste={'ON' if use_copy_paste else 'OFF'}")

