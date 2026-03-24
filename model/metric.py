import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = (torch.sigmoid(output) > 0.5).float()
        correct = torch.sum(pred == target).item()
    return correct / target.numel()

def iou(output, target, eps=1e-6):
    with torch.no_grad():
        pred = (torch.sigmoid(output) > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return ((intersection + eps) / (union + eps)).item()

def pct_low_accuracy_tiles(output, target, threshold=0.25):
    with torch.no_grad():
        pred = (torch.sigmoid(output) > 0.5).float()
        batch_size = target.shape[0]
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        per_tile_acc = (pred_flat == target_flat).float().mean(dim=1)
        low_acc_count = (per_tile_acc < threshold).sum().item()
    return low_acc_count / batch_size
