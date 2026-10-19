"""SAM and microSAM masks for the baseline arms, scored at native resolution.

Run before quantify_density.py: it writes masks/<arm>/<stem>.npz, which that
script's SAM and microSAM arms read.

Whole slide, not tiles -- SAM is an object segmenter and a 512 px tile is an
arbitrary crop of an object. Three things here are conclusions rather than
choices, and should not be re-litigated without redoing the sweep behind them:

  SIDE = 1024       SAM's encoder is fixed at img_size 1024; its pos_embed is
                    (1, 64, 64, 1280). This is its maximum supported resolution,
                    so the mask is upsampled ~34x to native against the model's
                    ~4.3x. That handicap is inherent to the comparison, not a
                    setting.
  --mask-logit 20   Mask-only prompting is unusable: swept over scale it ran
                    0.00 to 0.91 IoU on one slide and peaked somewhere different
                    on each of three, because SAM's prompt encoder was trained on
                    mask logits accompanied by a sparse prompt. With the interior
                    points in place it is flat across 12-30, so 20 is a midpoint
                    on a plateau, not a fitted value.
  --point-spacing 16  From spacing 32 inward IoU moves under 0.03 while the point
                    count changes thirtyfold. Also a plateau.

The prompt is roi_<stem>.pt whole and unaltered: (1, 256, 256) on the padded
square canvas, which is exactly SAM's dense mask-prompt frame, so it transfers
with no resampling. No box prompt -- these ROIs are arcs and a component fills a
median 28% of its own bounding box.

Intensity: SAM's API takes uint8 and has no learned gain, while our encoder takes
float32 and lets BatchNorm set it. One scalar linear gain per slide puts the
99.9th percentile over scanned pixels at 255 -- ONE scalar across R, G and B, as
per-channel would launder the real MYL4/counterstain imbalance -- applied in the
source dtype with the 8-bit cast last, since gaining after a full-scale
conversion recovers nothing.

    python figures-for-pub/quantification/export_sam_masks.py --arm stub-roi
    python figures-for-pub/quantification/export_sam_masks.py --arm sam
    python figures-for-pub/quantification/export_sam_masks.py --arm microsam

`--arm stub-roi` runs the same path with the ROI in place of the network, so its
IoU should reproduce roi-upsample.json's `roi` column. No GPU needed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'utils'))

from slide_level_metrics import score_native       # noqa: E402

SIDE = 1024        # SAM's img_size; its position embeddings are fixed at it
ROI_SIDE = 256     # SAM's dense-prompt frame, and the ROI's own grid
GAIN_PCT = 99.9    # percentile over scanned pixels mapped to 255

# arm -> (segment_anything architecture, weights file under --weights-dir).
# Both ViT-H so capacity is identical and only the finetuning differs. microSAM's
# light-microscopy generalist stops at ViT-L, so the histopathology generalist is
# the one that allows a same-capacity comparison.
ARMS = {
    'sam':      ('vit_h', 'sam_vit_h_4b8939.pth'),
    'microsam': ('vit_h', 'microsam_vit_h_histopathology.pt'),
    'stub-roi': (None, None),
}


def find(raw_dir, stem, kind):
    hits = sorted(Path(raw_dir).glob(f'*/{stem}_{kind}.*'))
    if not hits:
        raise SystemExit(f'no {kind} for {stem} under {raw_dir}')
    return hits[0]


def load_planes(path):
    """(R, G, B) at native resolution IN THE SOURCE DTYPE -- the gain is applied
    first and the 8-bit cast is the last step."""
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        arr, axes = series.asarray(), series.axes
    planes = [arr[c] for c in range(3)] if axes.startswith('C') \
        else [arr[..., c] for c in range(3)]
    for p in planes:
        if p.dtype not in (np.uint8, np.uint16):
            raise SystemExit(f'unexpected dtype {p.dtype} in {path}')
    return [np.ascontiguousarray(p) for p in planes]


def sam_frame(planes):
    """SAM's uint8 input at long side SIDE, how it was scaled, and the unpadded size.

    INTER_AREA because this is a ~34x reduction and a nearest or bilinear sample
    would alias away most of the section. The percentile is taken on the reduced
    frame, which is what SAM actually sees.

    `gain` is reported relative to mapping the dtype's FULL SCALE to 255, so 1.0
    is a plain 16-to-8-bit conversion and 14.2 means the section occupies a
    fourteenth of the range the file could hold. The raw multiplier would be an
    uninterpretable 0.0087 and is not comparable across source dtypes.
    """
    H, W = planes[0].shape
    full = float(np.iinfo(planes[0].dtype).max)
    s = SIDE / max(H, W)
    oh, ow = round(H * s), round(W * s)
    small = np.stack([cv2.resize(p, (ow, oh), interpolation=cv2.INTER_AREA)
                      for p in planes], -1)

    # unscanned canvas is pure black and would otherwise drag the percentile down
    scanned = small.max(-1) > 0
    if not scanned.any():
        raise SystemExit('frame is entirely unscanned')
    hi = float(np.percentile(small[scanned], GAIN_PCT))
    scale = 255.0 / hi if hi > 0 else 255.0 / full
    img = np.clip(small.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return img, {'gain': scale * full / 255.0, 'p_hi': hi, 'full_scale': full,
                 'dtype': str(planes[0].dtype),
                 'clipped_frac': float(np.count_nonzero(small > hi) / small.size)}, oh, ow


def load_roi(data_dir, stem):
    """The model's own ROI, raw. No threshold, no island filter, no resampling."""
    path = Path(data_dir) / 'images' / stem / f'roi_{stem}.pt'
    if not path.exists():
        raise SystemExit(f'no ROI for {stem} at {path}')
    roi = torch.load(path, weights_only=True).squeeze(0).numpy().astype(np.float32)
    if roi.shape != (ROI_SIDE, ROI_SIDE):
        raise SystemExit(f'{stem}: ROI is {roi.shape}, not ({ROI_SIDE}, {ROI_SIDE}) -- '
                         f'the correspondence with SAM\'s mask-prompt frame is what '
                         f'lets it be passed unresampled, so this is not resizable')
    return roi


def build_predictor(arm, weights_dir, device):
    from segment_anything import SamPredictor, sam_model_registry

    kind, fname = ARMS[arm]
    ckpt = Path(weights_dir) / fname
    if not ckpt.exists():
        raise SystemExit(f'{arm}: no weights at {ckpt}')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    # microSAM ships torch_em training checkpoints: the weights sit under
    # 'model_state' behind a 'sam.' prefix. Meta ships a bare state dict. Nothing
    # else differs, and load_state_dict is strict, so a wrong guess fails here
    # rather than loading something plausible.
    if isinstance(state, dict) and 'model_state' in state:
        state = {k[4:] if k.startswith('sam.') else k: v
                 for k, v in state['model_state'].items()}
    sam = sam_model_registry[kind]()
    sam.load_state_dict(state)
    return SamPredictor(sam.to(device).eval())


def prompt_points(roi, spacing):
    """Positive points over the ROI interior, in SAM frame pixels.

    A lattice at `spacing` ROI pixels, unioned with one point per connected
    component at its distance-transform maximum -- the maximum rather than the
    centroid because these components are arcs, and an arc's centroid can fall
    outside it. The per-component point is what keeps a component smaller than
    the lattice from being passed over in silence.
    """
    binary = (roi > 0.5).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, 8)
    s = SIDE / ROI_SIDE
    pts = []
    for i in range(1, n):
        d = cv2.distanceTransform((labels == i).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(d.argmax()), d.shape)
        pts.append((x * s, y * s))
    ys, xs = np.mgrid[0:ROI_SIDE:spacing, 0:ROI_SIDE:spacing]
    inside = binary[ys.ravel(), xs.ravel()] > 0
    pts += [(x * s, y * s) for x, y in zip(xs.ravel()[inside], ys.ravel()[inside])]
    if not pts:
        raise SystemExit('ROI is empty, so there is nothing to prompt with')
    return np.array(pts, np.float32)


def segment(predictor, img, roi, logit, spacing):
    """One SAM call for the whole section: the dense ROI plus its interior points.

    mask_input is logits, not 0/1: SAM's prompt encoder was trained on the mask
    logits of a previous iteration, so a 0/1 array sits far outside the range its
    downscaling convolutions ever saw.
    """
    pts = prompt_points(roi, spacing)
    predictor.set_image(img)                      # SAM pads to SIDE itself
    mask, score, _ = predictor.predict(
        point_coords=pts, point_labels=np.ones(len(pts), np.int32),
        mask_input=((roi - 0.5) * 2.0 * logit)[None],
        multimask_output=False)
    return mask[0], float(score[0]), len(pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = json.loads((ROOT / 'config.json').read_text())
    ap.add_argument('--arm', default='sam', choices=sorted(ARMS))
    ap.add_argument('--weights-dir', default=str(ROOT / 'saved' / 'sam'))
    ap.add_argument('--data-dir', default=str(ROOT / cfg['data_loader']['args']['data_dir']),
                    help='holds images/<stem>/roi_<stem>.pt, the prompt')
    ap.add_argument('--raw-dir', default=cfg['preprocess']['raw_image_directory'])
    ap.add_argument('--reference', default=str(ROOT / 'figures-for-pub' / 'roi-upsample'
                                               / 'roi-upsample.json'),
                    help='per-slide roi/model IoU to print against, and to verify '
                         'the stub-roi arm')
    ap.add_argument('--outdir', default=None, help='default <HERE>/masks/<arm>')
    ap.add_argument('--scores', default=None, help='default <HERE>/sam_scores.json')
    ap.add_argument('--mask-logit', type=float, default=20.0,
                    help='magnitude the 0/1 ROI is mapped to as SAM mask logits; '
                         'flat across 12-30, so a midpoint rather than a fit')
    ap.add_argument('--point-spacing', type=int, default=16,
                    help='lattice spacing in ROI pixels; IoU moves under 0.03 '
                         'between 32 and 4, so inside a plateau')
    ap.add_argument('--tol', type=float, default=0.01,
                    help='stub-roi only: largest IoU deviation from the reference '
                         'that still counts as reproducing it')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--slides', nargs='*', default=None)
    a = ap.parse_args()

    ref = {s['stem']: s for s in json.loads(Path(a.reference).read_text())['slides']} \
        if Path(a.reference).exists() else {}
    stems = a.slides or sorted(p.name for p in
                               (Path(a.data_dir) / 'images').iterdir() if p.is_dir())

    outdir = Path(a.outdir) if a.outdir else HERE / 'masks' / a.arm
    outdir.mkdir(parents=True, exist_ok=True)
    predictor = None if a.arm == 'stub-roi' \
        else build_predictor(a.arm, a.weights_dir, torch.device(a.device))

    rows = []
    for stem in stems:
        roi = load_roi(a.data_dir, stem)
        img, gain, oh, ow = sam_frame(load_planes(find(a.raw_dir, stem, 'image')))
        if img.shape[:2] != (oh, ow):
            raise SystemExit(f'{stem}: frame {img.shape[:2]} != ({oh}, {ow})')

        if predictor is None:
            # checkable against alternate code path to verify correct
            mask = cv2.resize(roi, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
            mask, score, n_pts = mask[:oh, :ow] > 0.5, float('nan'), 0
        else:
            mask, score, n_pts = segment(predictor, img, roi, a.mask_logit, a.point_spacing)

        # back onto the padded square, which is the frame score_native inverts
        padded = np.zeros((SIDE, SIDE), bool)
        padded[:oh, :ow] = mask
        # one whole-slide pass, so every pixel is covered and nothing is excused
        _, _, _, rec = score_native(padded, np.ones((SIDE, SIDE), bool), stem,
                                    a.raw_dir, save_to=outdir)

        r = ref.get(stem, {})
        rows.append({'stem': stem, 'arm': a.arm, **rec, **gain, 'sam_score': score,
                     'n_points': n_pts, 'frame_hw': [oh, ow],
                     'roi_reference': r.get('roi'), 'model_reference': r.get('model')})
        delta = '' if not r else f"   roi {r['roi']:.4f}   model {r['model']:.4f}"
        print(f"  {stem:<22} iou {rec['iou']:.4f}  dice {rec['dice']:.4f}"
              f"  gain {gain['gain']:5.2f}x {gain['dtype']:<8}"
              f"clip {gain['clipped_frac']*100:.2f}%  frame {oh}x{ow}{delta}")

    # merged, so running one arm does not erase another's scores
    scores = Path(a.scores) if a.scores else HERE / 'sam_scores.json'
    all_arms = json.loads(scores.read_text()) if scores.exists() else {}
    all_arms[a.arm] = {'gain_pct': GAIN_PCT, 'mask_logit': a.mask_logit,
                       'point_spacing': a.point_spacing, 'slides': rows}
    scores.write_text(json.dumps(all_arms, indent=2))

    v = np.array([r['iou'] for r in rows])
    print(f'\niou mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  '
          f'min {v.min():.4f}  max {v.max():.4f}  n={len(v)}')
    print(f'wrote {scores}\nwrote {len(rows)} masks to {outdir}')

    # geometry check against roi-upsample.json
    if a.arm == 'stub-roi':
        paired = [(r['stem'], abs(r['iou'] - r['roi_reference']))
                  for r in rows if r['roi_reference'] is not None]
        if not paired:
            raise SystemExit(f'no reference IoUs in {a.reference} to verify against')
        stem, worst = max(paired, key=lambda p: p[1])
        print(f'\nstub vs {Path(a.reference).name}: max deviation {worst:.4f} '
              f'on {stem}, over {len(paired)} slides (tolerance {a.tol})')
        if worst > a.tol:
            raise SystemExit('stub-roi does not reproduce the reference -- the frame, '
                             'the padding or the native inverse has moved. Masks are '
                             'written but should not be trusted.')
        print('geometry verified')


if __name__ == '__main__':
    main()
