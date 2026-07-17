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
        correct = torch.sum(pred == target).item()
    return correct / target.numel()

def iou(output, target, eps=1e-6):
    with torch.no_grad():
        output = _crop_to_target(output, target)
        pred = (torch.sigmoid(output) > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return ((intersection + eps) / (union + eps)).item()

