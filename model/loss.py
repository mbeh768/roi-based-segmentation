import torch
import torch.nn.functional as F


def bce(output, target):
    assert output.shape[-2:] == target.shape[-2:], \
        f"output {tuple(output.shape[-2:])} != target {tuple(target.shape[-2:])}"
    return F.binary_cross_entropy_with_logits(output, target)


def dice_loss(output, target, eps=1e-6):
    assert output.shape[-2:] == target.shape[-2:], \
        f"output {tuple(output.shape[-2:])} != target {tuple(target.shape[-2:])}"
    probs = torch.sigmoid(output)

    dims = (1, 2, 3)    # compute per image, then average over batch
    intersection = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)

    dice = 1 - (2 * intersection + eps) / (union + eps)
    return dice.mean()


def bce_dice(output, target, a=0.3):
    """Combined BCE + Dice."""
    return a * bce(output, target) + (1 - a) * dice_loss(output, target)
