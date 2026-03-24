import torch
import torch.nn.functional as F


def bce(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def dice_loss_batch(output, target, eps=1e-6):
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
    dice = dice_loss_batch(output, target)
    return a * _bce + (1 - a) * dice


def dice_boundary(output, target):
    dice = dice_loss_batch(output, target)
    boundary = boundary_loss(output, target)
    return 0.5 * dice + 0.5 * boundary


def boundary_loss(output, target):
    """Sobel-edge-weighted boundary loss."""
    device = output.device
    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
    ).unsqueeze(0).to(device)
    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
    ).unsqueeze(0).to(device)

    Ix = F.conv2d(target, sobel_x, padding=1)
    Iy = F.conv2d(target, sobel_y, padding=1)
    edge_target = torch.hypot(Ix, Iy)

    probs = torch.sigmoid(output)
    return torch.mean(edge_target * torch.abs(probs - target))
