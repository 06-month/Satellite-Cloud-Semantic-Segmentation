"""
Loss functions for Cloud Segmentation
"""
import torch
import torch.nn.functional as F


def ohem_ce_loss(preds, targets, ratio=0.25, ignore_index=255):
    """Online Hard Example Mining Cross-Entropy Loss"""
    targets = targets.long()
    if targets.dim() == 4:
        targets = targets.squeeze(1)

    n, c, h, w = preds.shape
    preds = preds.permute(0,2,3,1).reshape(-1, c)
    targets = targets.view(-1)

    ce = F.cross_entropy(preds, targets, ignore_index=ignore_index, reduction='none')
    ce_sorted, idx = torch.sort(ce, descending=True)

    num_hard = int(ce.numel() * ratio)
    ce_hard = ce_sorted[:num_hard]

    return ce_hard.mean()
    

def dice_loss(preds, targets, eps=1e-7):
    """
    Dice Loss
    preds: (B, C, H, W) - Logits
    targets: (B, H, W) or (B, 1, H, W) - Indices
    """
    # 1. Targets safe handling
    targets = targets.long()
    if targets.dim() == 4:
        targets = targets.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
    
    num_classes = preds.shape[1]
    
    # 2. One-hot Encoding
    true_1_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    
    # 3. Softmax
    probas = F.softmax(preds, dim=1)
    
    # 4. Calculate Dice
    dims = (0,) + tuple(range(2, targets.ndimension() + 1))
    
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    
    return 1 - (2. * intersection / (cardinality + eps)).mean()


def jaccard_loss(preds, targets, eps=1e-7):
    """Jaccard (IoU) Loss"""
    targets = targets.long()
    if targets.dim() == 4:
        targets = targets.squeeze(1)
        
    num_classes = preds.shape[1]
    true_1_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    probas = F.softmax(preds, dim=1)
    
    dims = (0,) + tuple(range(2, targets.ndimension() + 1))
    
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    
    return 1 - (intersection / (cardinality - intersection + eps)).mean()


def ce_loss(preds, targets, ignore=255):
    """Cross-Entropy Loss"""
    targets = targets.long()
    if targets.dim() == 4:
        targets = targets.squeeze(1)
        
    return F.cross_entropy(preds, targets, ignore_index=ignore)


def get_loss_function(name):
    """Get loss function by name"""
    if name == 'dice':
        return dice_loss
    elif name == 'jaccard':
        return jaccard_loss
    elif name == 'ce':
        return ce_loss
    elif name in ['dice+ce']:
        def loss(preds, targets):
            return 0.7 * ce_loss(preds, targets) + 0.3 * dice_loss(preds, targets)
        return loss
    elif name in ['dice+jaccard', 'dice + jaccard']:
        def combined_loss(preds, targets):
            return dice_loss(preds, targets) + jaccard_loss(preds, targets)
        return combined_loss
    elif name in ['ohem+dice']:
        def loss(preds, targets):
            return 0.7 * ohem_ce_loss(preds, targets, ratio=0.25) + \
                   0.3 * dice_loss(preds, targets)
        return loss
    else:
        print(f"Warning: Unknown loss name '{name}'. Using Dice Loss.")
        return dice_loss

