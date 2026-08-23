"""Preview the ROI perturbation used to simulate inter-annotator variation.

The knob is _roi_grid_distort (data_loader/datasets.py), a random grid warp --
NOT roi_jitter, which replaces the ROI with all-ones and so ablates the
constraint rather than varying it.

`strength` is a displacement in normalised grid coordinates and is not directly
interpretable, so every panel is labelled with the IoU between the perturbed ROI
and the original. That is the quantity a reader understands: published
inter-observer IoU for anatomical structures typically sits around 0.85-0.95, so
the sweep can be anchored against real annotator disagreement rather than an
arbitrary parameter.

Usage:
    python utils/roi_perturbation_preview.py --data-dir <dir> --stem <stem> --out preview.png
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader.datasets import _roi_grid_distort


def roi_iou(a, b):
    a = a > 0.5; b = b > 0.5
    inter = (a & b).sum().item(); union = (a | b).sum().item()
    return (inter + 1e-9) / (union + 1e-9)


def boundary_shift(a, b):
    """Symmetric mean surface distance, in ROI pixels.

    Measured both directions and averaged: a one-sided version only sees
    displacement that moves the boundary OUTWARD, and reads ~0 for a boundary
    that moved inward. For a thin branching structure like this ROI, IoU
    collapses under sub-pixel shifts because perimeter-to-area is huge, so
    surface distance is the more honest axis.
    """
    from scipy import ndimage
    a = (a > 0.5).numpy(); b = (b > 0.5).numpy()
    if not a.any() or not b.any():
        return float('nan')
    ea = a ^ ndimage.binary_erosion(a)
    eb = b ^ ndimage.binary_erosion(b)
    if not ea.any() or not eb.any():
        return float('nan')
    d_ab = ndimage.distance_transform_edt(~ea)[eb].mean()
    d_ba = ndimage.distance_transform_edt(~eb)[ea].mean()
    return float((d_ab + d_ba) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--stem', default=None)
    ap.add_argument('--strengths', default='0.0,0.01,0.02,0.035,0.05,0.08')
    ap.add_argument('--seeds', type=int, default=3, help='draws per strength')
    ap.add_argument('--out', default='roi_perturbation_preview.png')
    a = ap.parse_args()

    img_dir = Path(a.data_dir) / 'images'
    stem = a.stem or sorted(p.name for p in img_dir.iterdir() if p.is_dir())[0]
    glob_img = torch.load(img_dir / stem / 'global.pt', weights_only=True)[0]
    roi = torch.load(img_dir / stem / f'roi_{stem}.pt', weights_only=True)
    strengths = [float(s) for s in a.strengths.split(',')]

    g = glob_img.numpy()
    g = (g - g.min()) / (np.ptp(g) + 1e-9)   # ndarray.ptp removed in NumPy 2.0

    fig, axes = plt.subplots(a.seeds, len(strengths),
                             figsize=(2.5 * len(strengths), 2.6 * a.seeds))
    axes = np.atleast_2d(axes)
    stats = {}
    for col, s in enumerate(strengths):
        ious, shifts = [], []
        for row in range(a.seeds):
            torch.manual_seed(1000 + row); random.seed(1000 + row)
            pert = roi if s == 0 else _roi_grid_distort(roi.clone(), strength=s)
            i = roi_iou(roi[0], pert[0]); d = boundary_shift(roi[0], pert[0])
            ious.append(i); shifts.append(d)
            ax = axes[row, col]
            ax.imshow(g, cmap='gray', interpolation='nearest')
            ax.contour(roi[0].numpy(), levels=[0.5], colors='#4a7ba7', linewidths=1.4)
            ax.contour(pert[0].numpy(), levels=[0.5], colors='#c0504d', linewidths=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"strength {s}\nROI IoU {i:.3f}", fontsize=9)
            else:
                ax.set_title(f"IoU {i:.3f}", fontsize=8)
        stats[s] = (float(np.mean(ious)), float(np.std(ious)),
                    float(np.nanmean(shifts)))
    fig.suptitle(f"ROI perturbation preview — {stem}   "
                 f"(blue = original annotation, red = simulated second annotator)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}\n")
    print("%9s %12s %8s %14s %16s" % ("strength", "ROI IoU", "sd",
          "surf dist px", "at slide scale"))
    for s, (m, sd, sh) in stats.items():
        # ROI/global is 256 across a tiling space of 8192 -> 32x
        print("%9.3f %12.4f %8.4f %14.2f %13.0f px" % (s, m, sd, sh, sh * 32))
    json.dump({str(k): v for k, v in stats.items()},
              open(Path(a.out).with_suffix('.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
