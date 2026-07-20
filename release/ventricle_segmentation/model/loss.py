import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _crop_to_target(output, target):
    """If output is spatially larger than target, center-crop output to match.
    Lets the model see a context window wider than the loss/stitch region."""
    if output.shape[-2:] == target.shape[-2:]:
        return output
    oh, ow = output.shape[-2:]
    th, tw = target.shape[-2:]
    y0 = (oh - th) // 2
    x0 = (ow - tw) // 2
    return output[..., y0:y0 + th, x0:x0 + tw]


def bce(output, target):
    return F.binary_cross_entropy_with_logits(_crop_to_target(output, target), target)


def dice_loss(output, target, eps=1e-6):
    output = _crop_to_target(output, target)
    probs = torch.sigmoid(output)

    dims = (1, 2, 3)    # compute per image, then average over batch
    intersection = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)

    dice = 1 - (2 * intersection + eps) / (union + eps)
    return dice.mean()


def bce_dice(output, target, a=0.3):
    """Combined BCE + Dice. Output may be larger than target (center-cropped automatically)."""
    return a * bce(output, target) + (1 - a) * dice_loss(output, target)
