import torch


def accuracy(output, target):
    with torch.no_grad():
        assert output.shape[-2:] == target.shape[-2:], \
            f"output {tuple(output.shape[-2:])} != target {tuple(target.shape[-2:])}"
        pred = (torch.sigmoid(output) > 0.5).float()
        correct = torch.sum(pred == target).item()
    return correct / target.numel()


def iou(output, target, eps=1e-6):
    """Mean per-image IoU over the batch.

    dims=(1, 2, 3) reduces each image separately before averaging, matching
    dice_loss. Summing over the whole batch instead -- as this did previously --
    makes the result depend on batch size and on which tiles happen to share a
    batch, because a few large-foreground tiles then dominate the pooled union
    and the rest contribute almost nothing. Two runs of the same model at
    different batch sizes were not comparable.

    EMPTY TILES SCORE 1.0. A tile with no ground truth where nothing is
    predicted gives (0+eps)/(0+eps) = 1, i.e. counted as a perfect call, where
    the pooled form contributed nothing to either sum. That is the correct
    reading of "got it right" per image, but it means the metric is sensitive to
    how many empty tiles the sampling admits: with roi_exclusion about 6% of
    tiles are empty, without it around 77% are, so the two are not comparable.
    """
    with torch.no_grad():
        assert output.shape[-2:] == target.shape[-2:], \
            f"output {tuple(output.shape[-2:])} != target {tuple(target.shape[-2:])}"
        pred = (torch.sigmoid(output) > 0.5).float()
        dims = (1, 2, 3)
        intersection = (pred * target).sum(dims)
        union = pred.sum(dims) + target.sum(dims) - intersection
        return ((intersection + eps) / (union + eps)).mean().item()
