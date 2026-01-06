"""
Evaluation metrics for Cloud Segmentation
"""
import torch
import torch.nn.functional as F


def fitness_test(true, pred, num_classes=4):
    """Calculate mIOU, Pixel Accuracy, Dice Score"""
    eps = 1e-7
    
    # Safe handling of dimensions and types
    true = true.long()
    if true.dim() == 4:
        true = true.squeeze(1)
        
    # Pred: (B, C, H, W) -> (B, H, W) (argmax)
    if pred.shape[1] > 1:  # If logits
        pred_max = pred.argmax(dim=1)
    else:
        pred_max = pred.squeeze(1)  # Already argmax
    
    # Pixel Accuracy
    pix_acc = (true == pred_max).float().mean()
    
    # One-hot conversion for IoU/Dice calculation
    true_1h = F.one_hot(true, num_classes).permute(0, 3, 1, 2).float()
    pred_1h = F.one_hot(pred_max, num_classes).permute(0, 3, 1, 2).float()
    
    dims = (0,) + tuple(range(2, true.ndimension() + 1))
    
    # Intersection & Union
    inter = torch.sum(pred_1h * true_1h, dims)
    union = torch.sum(pred_1h + true_1h, dims) - inter
    
    # mIoU
    m_iou = (inter / (union + eps)).mean()
    
    # Dice Score
    dice = (2. * inter / (torch.sum(pred_1h + true_1h, dims) + eps)).mean()
    
    return m_iou.item(), pix_acc.item(), dice.item()

