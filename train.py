"""
Training script for Cloud Segmentation - CMX Model
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
import config
from models import create_model, count_parameters
from data import CloudDataset
from utils import get_loss_function, fitness_test, init_seeds


def visualize_predictions(model, val_dl, device, save_dir, epoch, num_samples=5):
    """Visualize predictions on validation set"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Denormalization constants
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    NIR_MEAN = 0.5
    NIR_STD = 0.25

    count = 0
    with torch.no_grad():
        for imgs, targets in val_dl:
            rgb, nir = imgs
            rgb, nir = rgb.to(device), nir.to(device)

            if targets.dim() == 4:
                targets = targets.squeeze(1)
            gt_batch = targets.long().cpu().numpy()

            preds = model(rgb, nir)
            if preds.shape[-2:] != targets.shape[-2:]:
                preds = F.interpolate(preds, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            
            pred_batch = preds.argmax(dim=1).cpu().numpy()

            for i in range(rgb.shape[0]):
                if count >= num_samples:
                    return

                # Denormalize images
                img_rgb = rgb[i].permute(1, 2, 0).cpu().numpy()
                img_rgb = img_rgb * STD + MEAN
                img_rgb = np.clip(img_rgb, 0, 1)

                img_nir = nir[i].squeeze().cpu().numpy()
                img_nir = img_nir * NIR_STD + NIR_MEAN
                img_nir = np.clip(img_nir, 0, 1)

                img_gt = gt_batch[i]
                img_pred = pred_batch[i]

                # Plot
                fig, ax = plt.subplots(1, 4, figsize=(16, 4))
                
                ax[0].imshow(img_rgb)
                ax[0].set_title("RGB")
                ax[0].axis("off")

                ax[1].imshow(img_nir, cmap='gray')
                ax[1].set_title("NIR")
                ax[1].axis("off")

                ax[2].imshow(img_pred, cmap='jet', vmin=0, vmax=3)
                ax[2].set_title("Prediction")
                ax[2].axis("off")

                ax[3].imshow(img_gt, cmap='jet', vmin=0, vmax=3)
                ax[3].set_title("GT Mask")
                ax[3].axis("off")

                plt.suptitle(f"Epoch {epoch} - Sample {count+1}", fontsize=14)
                plt.tight_layout()

                save_name = os.path.join(save_dir, f"epoch_{epoch}_sample_{count}.png")
                plt.savefig(save_name)
                plt.close()
                
                count += 1


def train_one_epoch(model, optimizer, data_loader, loss_fn, device, epoch, num_epochs, accumulation_steps=1):
    """Train for one epoch"""
    model.train()
    losses = []
    
    optimizer.zero_grad()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}/{num_epochs}')
    
    for i, (imgs, targets) in pbar:
        rgb, nir = imgs 
        rgb, nir, targets = rgb.to(device), nir.to(device), targets.to(device)
        
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()
        
        preds = model(rgb, nir)

        # Resize predictions to match target size
        if preds.shape[2:] != targets.shape[1:]:
            preds = F.interpolate(preds, size=targets.shape[1:], mode='bilinear', align_corners=False)
        
        loss = loss_fn(preds, targets)
        
        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update parameters
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * accumulation_steps
        losses.append(current_loss)
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    return np.mean(losses)


def val_one_epoch(model, data_loader, device, epoch, num_epochs):
    """Validate for one epoch"""
    model.eval()
    metrics = {'miou': [], 'acc': [], 'dice': []}
    pbar = tqdm(data_loader, desc=f'Val {epoch}/{num_epochs}')
    
    for imgs, targets in pbar:
        rgb, nir = imgs
        rgb, nir, targets = rgb.to(device), nir.to(device), targets.to(device)
        
        if targets.dim() == 4: 
            targets = targets.squeeze(1)
        targets = targets.long()
        
        with torch.no_grad():
            preds = model(rgb, nir)
            if preds.shape[2:] != targets.shape[1:]:
                preds = F.interpolate(preds, size=targets.shape[1:], mode='bilinear', align_corners=False)
            
            m_iou, pix_acc, dice = fitness_test(targets.unsqueeze(1), preds)
            metrics['miou'].append(m_iou)
            metrics['acc'].append(pix_acc)
            metrics['dice'].append(dice)
            pbar.set_postfix({'mIOU': f'{m_iou:.4f}', 'Acc': f'{pix_acc:.4f}'})

    if len(metrics['miou']) == 0:
        return {'mIOU': 0, 'Accuracy': 0, 'Dice': 0}

    return {
        'mIOU': np.mean(metrics['miou']), 
        'Accuracy': np.mean(metrics['acc']), 
        'Dice': np.mean(metrics['dice'])
    }


def train(model, optimizer, train_dl, val_dl, loss_func, epochs, device, 
          use_scheduler=False, save_path='./ckpt', scheduler_type='cosine', 
          val_interval=5, accumulation_steps=1):
    """Main training loop"""
    torch.cuda.empty_cache()
    loss_fn = get_loss_function(loss_func)
    
    os.makedirs(save_path, exist_ok=True)
    visual_save_dir = os.path.join(save_path, "visuals")
    
    scheduler = None
    if use_scheduler:
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_fit = 0.0
    weight_file = os.path.join(save_path, 'cmx_best.pt')

    print(f"Start Training... Total Epochs: {epochs} (Val every {val_interval})")
    print(f"Gradient Accumulation: {accumulation_steps} steps (Effective Batch: {train_dl.batch_size * accumulation_steps})")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_dl, loss_fn, device, epoch, epochs, accumulation_steps)
        
        # Scheduler step
        if scheduler:
            if scheduler_type == "plateau":
                scheduler.step(train_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch}] Loss: {train_loss:.4f} | LR: {current_lr:.2e}")

        # Validation
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            print(f"    >>> Running Validation at Epoch {epoch}...")
            val_metrics = val_one_epoch(model, val_dl, device, epoch, epochs)
            
            print(f"    Val mIOU: {val_metrics['mIOU']:.4f} | Acc: {val_metrics['Accuracy']:.4f}")

            # Visualize samples
            visualize_predictions(model, val_dl, device, visual_save_dir, epoch, num_samples=5)

            # Save best model
            current_score = val_metrics['mIOU']
            if current_score > best_fit:
                best_fit = current_score
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': val_metrics,
                }, weight_file)
                print(f"    >>> Best Model Saved! (mIOU: {best_fit:.4f})")
        
        print("-" * 50)

    print(f"Training Completed. Best mIOU: {best_fit:.4f}")
    return best_fit


def load_data():
    """Load data paths"""
    rgb_path = os.path.join(config.workspace_path, 'train/rgb/')
    ngr_path = os.path.join(config.workspace_path, 'train/ngr/')
    label_path = os.path.join(config.workspace_path, 'train/label/')
    
    rgb_images = sorted([os.path.join(rgb_path, x) for x in os.listdir(rgb_path)])
    ngr_images = sorted([os.path.join(ngr_path, x) for x in os.listdir(ngr_path)])
    label_images = sorted([os.path.join(label_path, x) for x in os.listdir(label_path)])
    
    return rgb_images, ngr_images, label_images


def create_dataloaders(rgb_images, ngr_images, label_images):
    """Create train and validation dataloaders"""
    train_split = int(len(rgb_images) * config.train_data_rate)
    
    # Train dataset
    train_ds = CloudDataset(
        rgb_paths=rgb_images[:train_split], 
        ngr_paths=ngr_images[:train_split], 
        label_paths=label_images[:train_split],
        is_train=True, 
        crop_size=config.patch_size, 
        use_copy_paste=config.use_copy_paste
    )
    
    # Validation dataset
    val_ds = CloudDataset(
        rgb_paths=rgb_images[train_split:], 
        ngr_paths=ngr_images[train_split:], 
        label_paths=label_images[train_split:],
        is_train=False
    )
    
    # DataLoaders
    train_dl = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    
    return train_dl, val_dl


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train CMX Model for Cloud Segmentation')
    parser.add_argument('--workspace', type=str, default=config.workspace_path, help='Dataset path')
    parser.add_argument('--output', type=str, default=config.output_path, help='Output path')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--backbone', type=str, default=config.cmx_backbone, help='Backbone model')
    parser.add_argument('--seed', type=int, default=config.seed, help='Random seed')
    args = parser.parse_args()
    
    # Update config
    config.workspace_path = args.workspace
    config.output_path = args.output
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.cmx_backbone = args.backbone
    config.seed = args.seed
    
    # Initialize seeds
    init_seeds(config.seed)
    
    # Load data
    print("Loading data...")
    rgb_images, ngr_images, label_images = load_data()
    print(f"  RGB: {len(rgb_images)}, NGR: {len(ngr_images)}, Label: {len(label_images)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dl, val_dl = create_dataloaders(rgb_images, ngr_images, label_images)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        config.device, 
        config.num_classes, 
        config.cmx_backbone, 
        config.cmx_pretrained, 
        config.use_timm_pretrained
    )
    print(f"  Backbone: {config.cmx_backbone}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.lr_backbone},
        {"params": model.decode_head.parameters(), "lr": config.lr_head},
    ], weight_decay=config.weight_decay)
    
    print(f"  Optimizer: AdamW with separated LR")
    print(f"   - Backbone LR: {config.lr_backbone:.1e}")
    print(f"   - Head LR:     {config.lr_head:.1e}")
    
    # Train
    print("\nStarting training...")
    save_path = os.path.join(config.output_path, 'ckpt')
    best_metrics = train(
        model=model, 
        optimizer=optimizer, 
        train_dl=train_dl, 
        val_dl=val_dl, 
        loss_func=config.loss_func, 
        epochs=config.epochs, 
        device=config.device,
        use_scheduler=config.use_scheduler,
        save_path=save_path,
        scheduler_type=config.scheduler_type,
        val_interval=config.val_interval,
        accumulation_steps=config.accumulation_steps
    )
    
    print(f"\nTraining completed!")
    print(f"Best mIOU: {best_metrics:.4f}")


if __name__ == '__main__':
    main()

