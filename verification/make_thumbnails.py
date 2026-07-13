"""Generate small comparison thumbnails (GT vs predicted subendocardium overlay,
each composited onto a checkerboard so the alpha channel is visible) for the
verification report. Keeps the PDF small; full-resolution originals are cited
by path/checksum in the report text instead of embedded.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
OUT = Path(__file__).parent / "thumbnails"
OUT.mkdir(exist_ok=True)


def checker_composite(png_path, size=(360, 520)):
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    checker = (np.indices((h, w)).sum(axis=0) // 14) % 2
    canvas = np.repeat(np.where(checker[..., None] == 0, 235, 210).astype(np.uint8), 3, axis=2)
    alpha = small[..., 3:4].astype(np.float32) / 255.0
    color = small[..., 0:3].astype(np.float32)
    out = (canvas.astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def make_pair_thumbnails(pair, stem_dirs):
    """stem_dirs: dict stem -> (gt_bands_dir, pred_bands_dir)"""
    for stem, (gt_dir, pred_dir) in stem_dirs.items():
        gt_png = REPO / gt_dir / "subendocardium.png"
        pred_png = REPO / pred_dir / "subendocardium.png"
        gt_img = checker_composite(gt_png)
        pred_img = checker_composite(pred_png)
        if gt_img is None or pred_img is None:
            print(f"  skip {stem}: missing PNG(s)")
            continue
        label_h = 30
        canvas = np.full((gt_img.shape[0] + label_h, gt_img.shape[1] * 2 + 10, 3), 255, np.uint8)
        cv2.putText(canvas, "Ground Truth", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Model Prediction (held-out)", (gt_img.shape[1] + 20, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[label_h:, :gt_img.shape[1]] = gt_img
        canvas[label_h:, gt_img.shape[1] + 10:] = pred_img
        out_path = OUT / f"{pair}_{stem}_gt_vs_pred.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    make_pair_thumbnails("AJK056-10", {
        "AJKV056-10-1-of-2": ("pairs_quant_gt/AJK056-10_pair/bands_1of2",
                              "held_out_results/AJK056-10_pair/quantification_pred"),
        "AJK056-10-2of2": ("pairs_quant_gt/AJK056-10_pair/bands_2of2",
                           "held_out_results/AJK056-10_pair/quantification_2of2_pred"),
    })
    make_pair_thumbnails("AJKY121-9", {
        "AJKY121-9-1": ("pairs_quant_gt/AJKY121-9_pair/bands_1of2",
                        "held_out_results/AJKY121-9_pair/quantification_pred"),
        "AJKY121-9-2": ("pairs_quant_gt/AJKY121-9_pair/bands_2of2",
                        "held_out_results/AJKY121-9_pair/quantification_2of2_pred"),
    })
    make_pair_thumbnails("AJKY121-8", {
        "AJKY121-8-1": ("pairs_quant_gt/AJKY121-8_pair/bands_1of2",
                        "held_out_results/AJKY121-8_pair/quantification_pred"),
        "AJKY121-8-2": ("pairs_quant_gt/AJKY121-8_pair/bands_2of2",
                        "held_out_results/AJKY121-8_pair/quantification_2of2_pred"),
    })
