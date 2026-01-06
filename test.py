"""
Testing and submission script for Cloud Segmentation - CMX Model
"""
import os
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
import config
from models import create_model
from data import CloudDataset
from utils import mask2rle


def load_best_model(model, device, checkpoint_path):
    """Load best model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Model not found at {checkpoint_path}")
        return model
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print(f'Model loaded from {checkpoint_path}')
    
    if 'best_score' in checkpoint:
        print(f"Best scores from training:")
        scores = checkpoint['best_score']
        if isinstance(scores, dict):
            for k, v in scores.items():
                print(f"  {k}: {v:.4f}")
    
    return model


def predict_test_set(model, test_dataloader, device, result_path):
    """Generate predictions on test set"""
    model.eval()
    os.makedirs(result_path, exist_ok=True)
    
    print(f"Generating predictions to {result_path}...")
    
    with torch.no_grad():
        for batch_idx, (imgs, img_paths) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            rgb, nir = imgs
            rgb, nir = rgb.to(device), nir.to(device)
            
            # Model inference
            preds = model(rgb, nir)
            
            # Restore original size
            original_img = cv2.imread(img_paths[0])
            h_orig, w_orig = original_img.shape[:2]
            
            if preds.shape[2:] != (h_orig, w_orig):
                preds = F.interpolate(preds, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            
            # Argmax to get class indices
            _, idx_mask = preds.max(1)
            
            # Process batch
            for i in range(rgb.shape[0]):
                # Create empty color image (H, W, 3)
                pred_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                
                # Convert GPU tensor to numpy
                mask = idx_mask[i].cpu().numpy()
                
                # Color mapping (BGR format, same as baseline)
                # 0: Background (Black) - [0, 0, 0] (already initialized to 0)
                pred_img[mask == 1] = [0, 0, 255]    # Thick Cloud (Red)
                pred_img[mask == 2] = [0, 255, 0]    # Thin Cloud (Green)
                pred_img[mask == 3] = [0, 255, 255]  # Cloud Shadow (Yellow)
                
                # Save image
                filename = os.path.basename(img_paths[i])
                cv2.imwrite(os.path.join(result_path, filename), pred_img)


def create_submission(result_path, output_path):
    """Create submission.csv from prediction results"""
    print("Creating submission.csv...")
    
    # Get sorted file list from results folder
    test_label_file_list = sorted(os.listdir(result_path))
    test_label_path_list = [os.path.join(result_path, x) for x in test_label_file_list]
    
    rle_list = []
    # Read each image and encode to RLE
    for file_path in tqdm(test_label_path_list, desc="RLE Encoding"):
        img = cv2.imread(file_path)  # Read as BGR
        rle = mask2rle(img)
        rle_list.append(rle)
    
    # Create DataFrame
    submission = pd.DataFrame({
        'Image_Label': test_label_file_list,
        'EncodedPixels': rle_list
    })
    
    submission_path = os.path.join(output_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"submission.csv saved to {submission_path}")
    print(submission.head())
    return submission


def create_test_dataloader(workspace_path):
    """Create test dataloader"""
    test_rgb_dir = os.path.join(workspace_path, 'test/rgb')
    test_ngr_dir = os.path.join(workspace_path, 'test/ngr')
    
    test_rgb = sorted([os.path.join(test_rgb_dir, x) for x in os.listdir(test_rgb_dir)])
    test_ngr = sorted([os.path.join(test_ngr_dir, x) for x in os.listdir(test_ngr_dir)])
    
    test_ds = CloudDataset(
        rgb_paths=test_rgb, 
        ngr_paths=test_ngr, 
        label_paths=[],  # No labels for test
        is_train=False
    )
    
    return DataLoader(
        test_ds, batch_size=1, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test CMX Model for Cloud Segmentation')
    parser.add_argument('--workspace', type=str, default=config.workspace_path, help='Dataset path')
    parser.add_argument('--output', type=str, default=config.output_path, help='Output path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--backbone', type=str, default=config.cmx_backbone, help='Backbone model')
    args = parser.parse_args()
    
    # Update config
    config.workspace_path = args.workspace
    config.output_path = args.output
    config.cmx_backbone = args.backbone
    
    # Set checkpoint path
    if args.checkpoint is None:
        checkpoint_path = os.path.join(config.output_path, 'ckpt', 'cmx_best.pt')
    else:
        checkpoint_path = args.checkpoint
    
    # Create model
    print("Creating model...")
    model = create_model(
        config.device, 
        config.num_classes, 
        config.cmx_backbone, 
        cmx_pretrained=None,
        use_timm_pretrained=False  # Load from checkpoint instead
    )
    
    # Load best model
    print(f"\nLoading model from checkpoint...")
    model = load_best_model(model, config.device, checkpoint_path)
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    test_dl = create_test_dataloader(config.workspace_path)
    print(f"Test batches: {len(test_dl)}")
    
    # Predict
    print("\nGenerating predictions...")
    result_path = os.path.join(config.output_path, 'results')
    predict_test_set(model, test_dl, config.device, result_path)
    
    # Create submission
    print("\nCreating submission file...")
    submission = create_submission(result_path, config.output_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

