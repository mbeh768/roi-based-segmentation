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


def _tversky_index(output, target, alpha, beta, eps=1e-6):
    """Per-image Tversky index in [0, 1] (1 = perfect). alpha weights false
    positives, beta false negatives; alpha=beta=0.5 is the Dice coefficient."""
    output = _crop_to_target(output, target)
    probs = torch.sigmoid(output)

    dims = (1, 2, 3)    # per image, then average over batch by the caller
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)

    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def tversky_loss(output, target, alpha=0.7, beta=0.3, eps=1e-6):
    """Tversky loss: a generalised Dice that weights false positives (alpha) and
    false negatives (beta) separately. alpha=beta=0.5 reduces to Dice; alpha>beta
    punishes over-segmentation (spurious foreground) harder, which is the lever
    for excluding tissue pockets the model wrongly includes."""
    return (1 - _tversky_index(output, target, alpha, beta, eps)).mean()


def focal_tversky_loss(output, target, alpha=0.7, beta=0.3, gamma=1.3333, eps=1e-6):
    """Focal Tversky: (1 - Tversky)^gamma. gamma>1 down-weights easy, well-covered
    tiles and concentrates gradient on hard, low-overlap ones — empirically better
    for small structures / thin boundaries. gamma=1 recovers plain tversky_loss."""
    return ((1 - _tversky_index(output, target, alpha, beta, eps)) ** gamma).mean()


def bce_tversky(output, target, a=0.3, alpha=0.7, beta=0.3):
    """BCE + Tversky, mirroring bce_dice. With alpha=beta=0.5 this IS bce_dice;
    alpha=0.7/beta=0.3 keeps the same BCE/region balance as 8M but makes the
    region term penalise false positives ~2.3x harder than misses."""
    return a * bce(output, target) + (1 - a) * tversky_loss(
        output, target, alpha=alpha, beta=beta)


def small_fp_loss(output, target, tau=8.0, overlap_thresh=0.5, eps=1e-6):
    """Penalise *small* spurious foreground blobs much more than large ones.

    For each connected component of the thresholded prediction whose overlap with
    the target is below overlap_thresh (i.e. a false-positive blob), every pixel
    gets weight exp(-size/tau): ~1 for a 1-2 px pocket, ~0 for a large structure.
    The loss is the weighted-mean predicted probability over those pixels, so the
    gradient pushes foreground down hardest in the smallest pockets and barely
    touches large/true regions. tau sets the size (px) below which pockets are
    heavily punished.

    Component labelling/sizing is done under no_grad (treated as constants);
    gradient flows only through the probabilities. This term has NO true-positive
    signal, so it must be added to an overlap loss (see bce_dice_pocket).

    NOTE: this targets ISOLATED small false-positive blobs. A pocket that is a
    filled hole *inside* the main foreground (no separating background ring) is
    part of the large component and is not penalised by this term.
    """
    output = _crop_to_target(output, target)
    probs = torch.sigmoid(output)
    with torch.no_grad():
        pred_bin = (probs > 0.5).cpu().numpy()
        tgt_bin = (target > 0.5).float().cpu().numpy()
        w = np.zeros(pred_bin.shape, dtype=np.float32)
        for b in range(pred_bin.shape[0]):
            lab, n = ndimage.label(pred_bin[b, 0])
            if n == 0:
                continue
            idx = np.arange(1, n + 1)
            sizes = ndimage.sum(np.ones(lab.shape, np.float32), lab, idx)
            tp = ndimage.sum(tgt_bin[b, 0], lab, idx)
            overlap = tp / (sizes + eps)
            weight_per_label = np.where(
                overlap < overlap_thresh, np.exp(-sizes / tau), 0.0
            ).astype(np.float32)
            lut = np.concatenate([[0.0], weight_per_label])  # label 0 = background
            w[b, 0] = lut[lab]
        weights = torch.from_numpy(w).to(probs.device)
    return (weights * probs).sum() / (weights.sum() + eps)


def bce_dice_pocket(output, target, a=0.3, pocket_weight=1.0, tau=8.0,
                    overlap_thresh=0.5):
    """bce_dice plus a small-false-positive-blob penalty (see small_fp_loss).
    pocket_weight scales how aggressively small spurious pockets are suppressed."""
    return bce_dice(output, target, a=a) + pocket_weight * small_fp_loss(
        output, target, tau=tau, overlap_thresh=overlap_thresh)


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

