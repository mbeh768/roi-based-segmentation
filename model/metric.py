import torch


def accuracy(output, target):
    with torch.no_grad():
        assert output.shape[-2:] == target.shape[-2:], \
            f"output {tuple(output.shape[-2:])} != target {tuple(target.shape[-2:])}"
        pred = (torch.sigmoid(output) > 0.5).float()
        correct = torch.sum(pred == target).item()
    return correct / target.numel()


def iou(output, target, eps=1e-6):
    with torch.no_grad():
        assert output.shape[-2:] == target.shape[-2:], \
            f"output {tuple(output.shape[-2:])} != target {tuple(target.shape[-2:])}"
        pred = (torch.sigmoid(output) > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return ((intersection + eps) / (union + eps)).item()
