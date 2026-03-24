import torch
import torch.nn.functional as F


def bce(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def dice_loss(output, target, eps=1e-6):
    probs = torch.sigmoid(output)

    dims = (1, 2, 3)    # compute per image, then average over batch
    intersection = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)

    dice = 1 - (2 * intersection + eps) / (union + eps)
    return dice.mean()


def bce_dice(output, target, a=0.5):
    """
    Combined Dice loss and Binary Cross Entropy loss.
    output: logits, shape (B, 1, H, W)
    target: binary mask, shape (B, 1, H, W)
    """
    _bce = F.binary_cross_entropy_with_logits(output, target)
    dice = dice_loss(output, target)
    return a * _bce + (1 - a) * dice

