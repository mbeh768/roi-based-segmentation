import torch
import torch.nn.functional as F


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


def mse(output, target):
    return F.mse_loss(output, target)


def weighted_mask_mse(output, target, mask, weight):
    """MSE where masked pixels are weighted more heavily.
    weight=1: masked pixels count the same as unmasked (plain MSE).
    weight=3: masked pixels contribute 3x more to the loss.
    mask: (B, 1, H, W) float {0, 1}, broadcasts across channels."""
    diff = (output - target) ** 2
    w = 1.0 + (weight - 1.0) * mask.expand_as(diff).to(diff.dtype)
    return (diff * w).sum() / w.sum()


def vicreg_features(feat, var_weight=1.0, cov_weight=0.04):
    """
    VICReg-style variance + covariance regularization on feature maps.

    Args:
        feat: (B, C, H, W) feature tensor
        var_weight: weight for variance term (encourages each channel to be active)
        cov_weight: weight for covariance term (encourages channels to be decorrelated)

    Returns:
        scalar loss
    """
    C = feat.shape[1]
    # flatten spatial dims: (B, C, H, W) -> (N, C)
    x = feat.permute(0, 2, 3, 1).reshape(-1, C)

    # variance: penalize channels whose std falls below 1
    std = torch.sqrt(x.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    # covariance: penalize off-diagonal correlations
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (x.shape[0] - 1)        # (C, C)
    off_diag = cov - torch.diag(cov.diag())
    cov_loss = (off_diag ** 2).sum() / C

    return var_weight * var_loss + cov_weight * cov_loss


def l1(output, target):
    return F.l1_loss(output, target)


def pixel_norm_mse(output, target, eps=1e-6):
    """
    MAE-style per-image normalised MSE: z-score both output and target along
    (C, H, W) per sample, then MSE. Removes the "match the mean intensity"
    shortcut that causes blurry reconstructions.
    """
    def _znorm(x):
        dims = (1, 2, 3)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        return (x - mean) / (std + eps)
    return F.mse_loss(_znorm(output), _znorm(target))


def dense_byol(z1, z2, t1, t2):
    """
    Symmetric per-pixel BYOL loss.

    Inputs are (B, C, H, W); we L2-normalize along channels at each spatial
    location, then compute 2 - 2 * cos(online_i, sg(target_j)) averaged over
    all pixels and the batch, symmetrized across the two view assignments.
    """
    def _one(a, b):
        a = F.normalize(a, dim=1)
        b = F.normalize(b.detach(), dim=1)
        return (2 - 2 * (a * b).sum(dim=1)).mean()
    return 0.5 * (_one(z1, t2) + _one(z2, t1))

