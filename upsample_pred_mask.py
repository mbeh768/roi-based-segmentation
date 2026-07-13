"""
Upsample a model prediction from the padded tiling-size square (e.g. 8192x8192,
zero-padded, aspect-ratio-preserved downsampled view -- see inference.py's
predict_prob_map) back to native slide resolution, matching the same
crop-then-resize logic pipeline.py uses at inference time.

Takes the continuous probability map (not an already-binarized mask) so
upsampling interpolates properly before a single clean threshold, rather than
upsampling a binary mask and re-blurring its edges.

Usage:
    python upsample_pred_mask.py --prob-map <prob_map.tif> \
        --native-h 22131 --native-w 14837 --out native_pred_mask.tif
    # or infer native dims from a same-slide reference file:
    python upsample_pred_mask.py --prob-map <prob_map.tif> \
        --reference <groundtruth_mask.tiff> --out native_pred_mask.tif
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


def main(args):
    prob = tifffile.imread(str(args.prob_map)).astype(np.float32)
    tsz = prob.shape[0]
    if prob.shape[0] != prob.shape[1]:
        raise ValueError(f"expected a square padded prob map, got {prob.shape}")

    if args.reference:
        ref = tifffile.imread(str(args.reference))
        if ref.ndim == 3:
            ref = ref[0] if ref.shape[0] in (1, 3) else ref[..., 0]
        native_h, native_w = ref.shape[:2]
    else:
        native_h, native_w = args.native_h, args.native_w
    if not native_h or not native_w:
        raise ValueError("pass --native-h/--native-w or --reference")

    scale = tsz / max(native_h, native_w)
    new_h, new_w = int(native_h * scale), int(native_w * scale)
    prob_crop = prob[:new_h, :new_w]

    prob_up = cv2.resize(prob_crop, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    mask = (prob_up > args.threshold).astype(np.uint8) * 255

    tifffile.imwrite(str(args.out), mask)
    print(f"{args.prob_map}: padded {tsz}x{tsz} -> content {new_h}x{new_w} -> "
          f"native {native_h}x{native_w}  foreground={int((mask>0).sum())}px "
          f"({100*(mask>0).mean():.2f}%)")
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prob-map', required=True, help='padded-square prob_map.tif from save_full_masks.py')
    parser.add_argument('--native-h', type=int, default=None)
    parser.add_argument('--native-w', type=int, default=None)
    parser.add_argument('--reference', default=None,
                        help='a same-slide file at native resolution to read H,W from '
                             '(e.g. the ground-truth mask)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
