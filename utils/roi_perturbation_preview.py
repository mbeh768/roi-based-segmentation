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
    """Mean distance from perturbed boundary pixels to the original boundary."""
    from scipy import ndimage
    a = (a > 0.5).numpy(); b = (b > 0.5).numpy()
    if not a.any() or not b.any():
        return float('nan')
    da = ndimage.distance_transform_edt(~a)
    eb = b ^ ndimage.binary_erosion(b)
    return float(da[eb].mean()) if eb.any() else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--stem', default=None)
    ap.add_argument('--strengths', default='0.0,0.05,0.1,0.2,0.3,0.5')
    ap.add_argument('--seeds', type=int, default=3, help='draws per strength')
    ap.add_argument('--out', default='roi_perturbation_preview.png')
    a = ap.parse_args()

    img_dir = Path(a.data_dir) / 'images'
    stem = a.stem or sorted(p.name for p in img_dir.iterdir() if p.is_dir())[0]
    glob_img = torch.load(img_dir / stem / 'global.pt', weights_only=True)[0]
    roi = torch.load(img_dir / stem / f'roi_{stem}.pt', weights_only=True)
    strengths = [float(s) for s in a.strengths.split(',')]

    g = glob_img.numpy()
    g = (g - g.min()) / (g.ptp() + 1e-9)

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
    print("%9s %12s %10s %16s" % ("strength", "ROI IoU", "sd", "boundary shift px"))
    for s, (m, sd, sh) in stats.items():
        print("%9.2f %12.4f %10.4f %16.1f" % (s, m, sd, sh))
    json.dump({str(k): v for k, v in stats.items()},
              open(Path(a.out).with_suffix('.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
