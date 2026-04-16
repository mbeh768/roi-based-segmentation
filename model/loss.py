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


def bce_dice(output, target, a=0.3):
    """
    Combined Dice loss and Binary Cross Entropy loss.
    output: logits, shape (B, 1, H, W)
    target: binary mask, shape (B, 1, H, W)
    """
    _bce = F.binary_cross_entropy_with_logits(output, target)
    dice = dice_loss(output, target)
    return a * _bce + (1 - a) * dice


def mse(output, target):
    return F.mse_loss(output, target)


def masked_mse(output, target, mask):
    """MSE restricted to pixels where mask > 0.
    mask: (B, 1, H, W) float {0, 1}, broadcasts across channels.
    Falls back to plain MSE when mask is empty."""
    diff = (output - target) ** 2
    mask_bchw = mask.expand_as(diff).to(diff.dtype)
    denom = mask_bchw.sum()
    if denom < 1.0:
        return diff.mean()
    return (diff * mask_bchw).sum() / denom


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

