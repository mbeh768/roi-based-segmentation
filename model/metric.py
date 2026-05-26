import torch


def _crop_to_target(output, target):
    if output.shape[-2:] == target.shape[-2:]:
        return output
    oh, ow = output.shape[-2:]
    th, tw = target.shape[-2:]
    y0 = (oh - th) // 2
    x0 = (ow - tw) // 2
    return output[..., y0:y0 + th, x0:x0 + tw]


def accuracy(output, target):
    with torch.no_grad():
        output = _crop_to_target(output, target)
        pred = (torch.sigmoid(output) > 0.5).float()
        valid = target >= 0
        if valid.all():
            correct = torch.sum(pred == target).item()
            return correct / target.numel()
        n = valid.sum().item()
        if n == 0:
            return 0.0
        correct = torch.sum((pred == target) & valid).item()
        return correct / n

def iou(output, target, eps=1e-6):
    with torch.no_grad():
        output = _crop_to_target(output, target)
        valid = (target >= 0).float()
        pred = (torch.sigmoid(output) > 0.5).float() * valid
        target = target.clamp(min=0) * valid
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


def psnr(output, target, max_val=5.0):
    with torch.no_grad():
        mse_val = torch.nn.functional.mse_loss(output, target)
        if mse_val < 1e-10:
            return 100.0
        return (10 * torch.log10(torch.tensor(max_val ** 2, device=mse_val.device) / mse_val)).item()
