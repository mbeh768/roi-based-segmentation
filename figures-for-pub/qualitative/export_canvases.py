"""Stitch and SAVE the held-out prediction canvases for the qualitative figure.

utils/slide_level_iou.py already stitches these canvases, but it keeps only
scalar summaries per slide and throws the arrays away. The qualitative figure
needs the arrays themselves, so this re-runs the same stitch and writes them to
disk. The stitching logic is deliberately identical -- accumulate onto an
image_H x image_W canvas, divide by the coverage count, threshold ONCE -- so the
IoU printed here can be checked against the number already in
figures-for-pub/baseline/k-fold-baseline_slide_iou.json. The script flags any
slide that disagrees by more than 1e-4.

Only the folds holding out a requested slide are loaded, so exporting four
slides costs four checkpoints rather than fourteen.

Output is one .npz per slide holding three boolean masks at full canvas
resolution -- pred, gt, covered -- compressed, because they are large and almost
entirely empty. 'covered' matters: pixels no tile reached are excluded from the
score rather than counted as correct background, and the figure has to mask them
out too or it would paint uncovered slide as agreement.

Usage:
    python figures-for-pub/qualitative/export_canvases.py
    python figures-for-pub/qualitative/export_canvases.py --device cpu
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# the repo root is the directory holding train.py, at whatever depth
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))

import data_loader.data_loaders as module_data   # noqa: E402
import model.model as module_arch                # noqa: E402

# Two best and two worst slides, each from a DISTINCT case, ordered best to
# worst. Four rather than six so each panel is large enough to read: at this
# figure width six pairs leaves every section a thumbnail.
SLIDES = [
    '21PControl-9-2of2',    # 0.9918
    '17P391_7A',            # 0.9877
    'AJKY121-7-2',          # 0.8844
    '17P645-1',             # 0.8691
]

# Loader settings that are an evaluation choice rather than a config choice.
# Dropped from the config args rather than copied and then overridden -- the
# loader's own defaults for hflip/vflip/rot90 are 0.0, so their absence is
# augmentation off, and the assert below confirms it.
EVAL_CONTROLLED = ('hflip', 'vflip', 'rot90', 'shuffle', 'num_workers', 'batch_size')


def load_raw_rgb(raw_dir, stem, target):
    """The stained section from the source TIFF, brightened to a mean of 127.

    These files carry no pyramid, so there is no cheap low-resolution level and
    the largest is 3 x 25975 x 35168 uint16 -- 5.5 GB in one piece. Two layouts
    appear: CYX, which stores one channel per page, and YXS, which is a single
    HWC page. The CYX case is read a page at a time and each plane shrunk before
    the next is touched, so peak memory is one 1.8 GB plane rather than the
    whole array. Resizing the uint16 in place rather than casting to float32
    first keeps it from doubling again.

    Returned at the section's OWN aspect ratio. Preprocessing zero-padded to a
    square because the tiler wanted one; nothing downstream does any more, and
    padding the figure would show the reader empty canvas that is not part of
    the specimen. The caller crops the masks to match.

    Brightness is a plain gain to a mean of 127, not a percentile stretch or
    per-channel standardisation: this panel shows the reader the raw input, so
    the only adjustment is the one that makes it visible on the page.

    The source folder names do not match the preprocessed stems -- AJKY121-7-2
    lives in AJKY121_7, and 17P645-1 in a folder with a parenthetical suffix --
    but the FILE names carry the stem, so the glob keys on those.
    """
    import tifffile

    hits = sorted(Path(raw_dir).glob(f'*/{stem}_image.*'))
    if not hits:
        raise SystemExit(f'no raw image for {stem} under {raw_dir}')
    if len(hits) > 1:
        print(f'    {stem}: {len(hits)} candidates, using {hits[0].name}')

    def dsize(H, W):
        scale = target / max(H, W)
        return max(1, round(W * scale)), max(1, round(H * scale))

    with tifffile.TiffFile(hits[0]) as tif:
        series = tif.series[0]
        if series.axes.startswith('C'):
            H, W = series.shape[1], series.shape[2]
            img = np.stack([cv2.resize(tif.pages[c].asarray(), dsize(H, W),
                                       interpolation=cv2.INTER_AREA)
                            for c in range(series.shape[0])], axis=-1)
        else:
            arr = series.asarray()
            H, W = arr.shape[:2]
            img = cv2.resize(arr, dsize(H, W), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32)
    img = np.clip(img * (127.0 / max(img.mean(), 1e-6)), 0, 255).astype(np.uint8)
    print(f'    {stem}: {W}x{H} {series.dtype} {series.axes} -> '
          f'{img.shape[1]}x{img.shape[0]}  mean {img.mean():.1f}')
    return img


def stitch(model, ds, idxs, wanted, device, amp, amp_dtype, batch_size):
    """Accumulate predictions onto per-slide canvases, for wanted stems only."""
    from torch.utils.data.dataloader import default_collate

    canvas = {}
    with torch.no_grad():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            batch = default_collate([ds[i] for i in chunk])
            *inputs, target = batch
            inputs = [t.to(device) for t in inputs]
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp):
                out = model(*inputs)
            prob = torch.sigmoid(out.float()).squeeze(1).cpu().numpy()
            gt = (target > 0.5).float().squeeze(1).numpy()

            for k, i in enumerate(chunk):
                _, stem, e = ds.items[i]
                if stem not in wanted:
                    continue
                if stem not in canvas:
                    H, W = e['image_H'], e['image_W']
                    canvas[stem] = {'p': np.zeros((H, W), np.float32),
                                    'g': np.zeros((H, W), np.float32),
                                    'n': np.zeros((H, W), np.uint16)}
                y, x, h, w = e['y'], e['x'], e['height'], e['width']
                c = canvas[stem]
                c['p'][y:y + h, x:x + w] += prob[k]
                c['g'][y:y + h, x:x + w] += gt[k]
                c['n'][y:y + h, x:x + w] += 1
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--name', default='k-fold-baseline')
    ap.add_argument('--models-dir', default=str(ROOT / 'saved' / 'models'))
    ap.add_argument('--outdir', default=str(HERE / 'canvases'))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--slides', nargs='*', default=SLIDES)
    ap.add_argument('--raw-dir',
                    default=json.loads((ROOT / 'config.json').read_text())
                    ['preprocess']['raw_image_directory'],
                    help='source TIFFs, for the stained section shown beside the map')
    ap.add_argument('--reference', default=str(ROOT / 'figures-for-pub' / 'baseline'
                                               / 'k-fold-baseline_slide_iou.json'),
                    help='slide-level JSON to cross-check the recomputed IoU against')
    a = ap.parse_args()

    wanted = set(a.slides)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ref = {}
    if Path(a.reference).exists():
        for f in json.loads(Path(a.reference).read_text())['folds']:
            ref.update({k: v['iou'] for k, v in f['per_slide'].items()})

    runs = sorted(Path(a.models_dir).glob(f'{a.name}-f*'))
    runs = [d for d in runs if (d / 'model_best.pth').exists() and (d / 'config.json').exists()]
    if not runs:
        raise SystemExit(f'no runs with a checkpoint under {a.models_dir}/{a.name}-f*')

    device = torch.device(a.device)
    found = set()

    for d in runs:
        cfg = json.loads((d / 'config.json').read_text())
        args = {k: v for k, v in cfg['data_loader']['args'].items()
                if k not in EVAL_CONTROLLED}
        args.update(shuffle=False, num_workers=0, batch_size=a.batch_size)
        loader = getattr(module_data, cfg['data_loader']['type'])(**args)
        ds, idxs = loader.dataset, list(loader.test_indices)
        assert not ds.augment, 'augmentation still on -- the export would be noisy'

        # skip the fold entirely unless it holds out something we asked for.
        # Checked from ds.items so no checkpoint is loaded for an unused fold.
        stems = {ds.items[i][1] for i in idxs}
        if not stems & wanted:
            continue

        model = getattr(module_arch, cfg['arch']['type'])(**cfg['arch']['args']).to(device)
        state = torch.load(d / 'model_best.pth', map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        model.eval()

        # the checkpoint was trained under amp, so inference matches training
        # numerically. torch.autocast('cuda') has nothing to act on when device is CPU.
        amp = bool(cfg['trainer'].get('amp', False)) and device.type == 'cuda'
        amp_dtype = getattr(torch, cfg['trainer'].get('amp_dtype', 'bfloat16'))
        canvas = stitch(model, ds, idxs, wanted, device, amp, amp_dtype, a.batch_size)

        for stem, c in canvas.items():
            covered = c['n'] > 0
            count = np.maximum(c['n'], 1)
            # average the overlapping predictions, THEN threshold once
            pred = (c['p'] / count > 0.5) & covered
            gt = (c['g'] / count > 0.5) & covered
            inter = int(np.count_nonzero(pred & gt))
            union = int(np.count_nonzero(pred | gt))
            iou = inter / union if union else float('nan')

            # crop the square canvas back to the section's own aspect. The pad
            # sits bottom-right and no tile ever covered it, so this leaves the
            # IoU above untouched -- it only stops the figure showing empty
            # canvas as though it were slide.
            rgb = load_raw_rgb(a.raw_dir, stem, pred.shape[0])
            h, w = rgb.shape[:2]
            assert h <= pred.shape[0] and w <= pred.shape[1], \
                f'{stem}: raw image {h}x{w} exceeds the {pred.shape} canvas'

            # the ROI the model was actually given: roi_<stem>.pt, no suffix.
            # 256x256 for the whole slide, so 32x coarser than the canvas --
            # nearest-neighbour up, because interpolating would invent a soft
            # edge the model never saw.
            roi = torch.load(ROOT / args['data_dir'] / 'images' / stem / f'roi_{stem}.pt',
                             weights_only=True).squeeze(0).numpy()
            roi = cv2.resize(roi, (pred.shape[1], pred.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0.5

            np.savez_compressed(outdir / f'{stem}.npz', pred=pred[:h, :w],
                                gt=gt[:h, :w], covered=covered[:h, :w],
                                roi=roi[:h, :w], image=rgb, iou=iou,
                                run=d.name, stem=stem)
            found.add(stem)

            flag = f'  !! reference says {ref[stem]:.4f}' \
                if stem in ref and abs(ref[stem] - iou) > 1e-4 else ''
            print(f'  {stem:<22} iou {iou:.4f}  {pred.shape[0]}x{pred.shape[1]}'
                  f'  covered {covered.mean()*100:.1f}%{flag}')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    missing = wanted - found
    if missing:
        print(f'\nWARNING not found in any fold: {sorted(missing)}')
    print(f'\nwrote {len(found)} canvases to {outdir}')


if __name__ == '__main__':
    main()
