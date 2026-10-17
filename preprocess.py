'''
Script to process dataset images into downsampled .pt files for global prediction.
Loads gigapixel TIFF (via OpenSlide) or CZI (via pylibCZIrw) images and saves
downsampled (C, H, W) float32 tensors to data_dir/images/ and data_dir/masks/.
'''

import argparse
from pathlib import Path
import json
import logging
import shutil
import sys
from glob import glob
import cv2
import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger.logger import setup_logging


class Preprocessor:
    def __init__(self, config):
        self.c = config

    def clear_existing_data(self):
        '''Remove and recreate data_dir/images/ and data_dir/masks/.'''
        data_dir = Path(self.c['data_loader']['args']['data_dir'])
        for subdir in ('images', 'masks'):
            d = data_dir / subdir
            if d.exists():
                print(f"removing existing {subdir}")
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        print("cleared existing data")

    def find_source_filepaths(self, names=None, nomask=False):
        '''
        Find all paths to images (and masks) in the dataset directory.

        Args:
            names (set|None): if provided, only return files whose stem is in this set
            nomask (bool): if True, skip mask search and return empty list for masks.
                           When nomask=True, the image glob is broadened to match ANY
                           .tif/.tiff/.czi file, not just files containing '_image' in
                           the name.
        '''
        dir = Path(self.c['preprocess']['raw_image_directory'])

        print(f"searching:\t{str(dir)}")

        if nomask:
            i_paths = sorted(
                glob(str(dir / "*/*.tif*")) +
                glob(str(dir / "*/*.czi"))
            )
            i_paths = [Path(f) for f in i_paths]
            if names is not None:
                i_paths = [p for p in i_paths if p.stem.replace('.ome', '') in names]
            print(f"found {len(i_paths)} image paths (no-mask mode)")
            return i_paths, []

        i_paths = sorted(
            glob(str(dir / "*/*_image*.tif*")) +
            glob(str(dir / "*/*_image*.czi"))
        )
        i_paths = [Path(f) for f in i_paths]
        if names is not None:
            i_paths = [p for p in i_paths if p.stem.replace('.ome', '') in names]

        m_paths = sorted(
            glob(str(dir / "*/*_mask*.tif*")) +
            glob(str(dir / "*/*_mask*.czi"))
        )
        m_paths = [Path(f) for f in m_paths]
        if names is not None:
            m_paths = [p for p in m_paths if p.stem.replace('.ome', '') in names]

        print(f"found {len(i_paths)} image paths, and {len(m_paths)} mask paths")
        return i_paths, m_paths

    def _load_tif_openslide(self, path, target_size):
        '''Load TIF/TIFF via OpenSlide at the best pyramid level for target_size.

        Reads at the closest pyramid level then returns CHW uint8 ndarray (3 ch, RGB).
        Raises openslide.OpenSlideUnsupportedFormatError if the file is not supported.
        '''
        import openslide

        slide = openslide.OpenSlide(str(path))
        W, H = slide.dimensions
        downsample_factor = max(W, H) / target_size
        level = slide.get_best_level_for_downsample(downsample_factor)
        lw, lh = slide.level_dimensions[level]

        print(f"\topenslide: {W}x{H} -> level {level} ({lw}x{lh})")

        region = slide.read_region((0, 0), level, (lw, lh))  # PIL RGBA
        img = np.array(region)[:, :, :3]                     # HWC RGB uint8
        return np.transpose(img, (2, 0, 1))                  # CHW uint8

    def _load_tif_tifffile(self, path):
        '''Load TIF/TIFF image as CHW array in original dtype (no conversion).'''
        with tifffile.TiffFile(path) as tif:
            image = tif.asarray()

        if image.ndim == 2:
            image = image[np.newaxis, :, :]          # HW -> CHW
        elif image.ndim == 3 and np.argmin(image.shape) == 2:
            image = np.transpose(image, (2, 0, 1))   # HWC -> CHW

        return image                                  # CHW original dtype

    def _load_czi(self, path, scene_index=0, zoom=None):
        '''Load CZI image as CHW array, zoomed so pixel count doesn't exceed czi_max_px.

        Args:
            scene_index (int): index into czidoc.scenes_bounding_rectangle.
                Defaults to 0; pass via --namescsv second column to pick a specific scene.
            zoom (float|None): explicit pyramid zoom factor in (0, 1]. When given, the
                czi_max_px halving loop is skipped and the image is read at this zoom.
                When None (default), zoom is chosen automatically so the pixel count
                doesn't exceed config['preprocess']['czi_max_px'].

        Requires: pylibCZIrw  (pip install pylibCZIrw)
        '''
        from pylibCZIrw import czi as pyczi

        _max_czi_px = self.c['preprocess']['czi_max_px']
        with pyczi.open_czi(str(path)) as czidoc:
            bb = czidoc.total_bounding_box
            H = bb['Y'][1] - bb['Y'][0]
            W = bb['X'][1] - bb['X'][0]
            n_channels = bb['C'][1] - bb['C'][0] if 'C' in bb else 1
            if zoom is None:
                zoom = 1.0
                while zoom > 0.0 and (H * zoom) * (W * zoom) > _max_czi_px:
                    zoom *= 0.5
            print(f"\tczi image dimensions: {H}x{W} ({H*W:.2e} px), {n_channels} channel(s), pyramid zoom: {zoom} -> {int(H*zoom)}x{int(W*zoom)} ({H*zoom*W*zoom:.2e} px)\n")
            scenes = list(czidoc.scenes_bounding_rectangle.keys())
            if scenes:
                if scene_index >= len(scenes):
                    raise IndexError(
                        f"{path.name}: scene_index {scene_index} out of range "
                        f"({len(scenes)} scene(s) available)"
                    )
                scene = scenes[scene_index]
                print(f"\tczi scene: index {scene_index} -> {scene} (of {len(scenes)})")
            else:
                scene = None

            # read each channel separately and stack — read() without
            # a plane coordinate defaults to C=0 only
            channels = []
            for c in range(n_channels):
                plane = czidoc.read(plane={'C': c}, scene=scene, zoom=zoom)
                if plane.ndim == 3:
                    channels.append(plane)            # (H, W, 1) or (H, W, >=1)
                else:
                    channels.append(plane[:, :, np.newaxis])  # (H, W) -> (H, W, 1)
            data = np.concatenate(channels, axis=2)   # (H, W, total_C)

        C = data.shape[2]
        print(f"\tczi loaded: shape {data.shape} ({C} channel{'s' if C != 1 else ''})")

        data = np.transpose(data, (2, 0, 1))              # CHW

        return data                                        # CHW original dtype

    def load_and_format(self, path, interpolation, is_mask=False, target_size=None, scene_index=0):
        '''
        Load an image or mask, resize to a square, normalize to float32 [0, 1],
        zero-pad to square.

        The dtype scaling below is the only normalization applied to the input;
        everything after it is handled by the encoder's BatchNorm.

        Args:
            path (str|Path): path to image or mask file
            interpolation: cv2 interpolation flag for resizing
            is_mask (bool): if True, use tifffile loader (preserves label values)
            target_size (int|None): override for output square size;
                                    defaults to config['preprocess']['global_downsample']

        Returns:
            np.ndarray: shape (C, H, W) float32, where C=3 for RGB images, C=1 for masks
        '''
        path = Path(path)
        suffix = path.suffix.lower()
        _target = target_size if target_size is not None else self.c['preprocess']['global_downsample']

        if suffix == '.czi':
            image = self._load_czi(path, scene_index=scene_index)
        elif is_mask:
            image = self._load_tif_tifffile(path)
        else:
            try:
                image = self._load_tif_openslide(path, _target)
            except Exception as e:
                print(f"\topenslide failed ({type(e).__name__}), falling back to tifffile")
                image = self._load_tif_tifffile(path)

        # enforce 3-channel images (CHW at this point); skip if collapse will reduce to 1ch
        collapse = self.c['preprocess'].get('collapse', False)
        if not is_mask and not collapse and image.shape[0] != 3:
            handle = self.c['preprocess'].get('handle_missing_or_additional_channels', False)
            if not handle:
                raise ValueError(
                    f"{path.name}: expected 3 channels, got {image.shape[0]}. "
                    f"Set preprocess.handle_missing_or_additional_channels=true to auto-fix."
                )
            C = image.shape[0]
            if C > 3:
                # drop channels with lowest mean until 3 remain
                means = [image[c].mean() for c in range(C)]
                keep = sorted(range(C), key=lambda c: means[c], reverse=True)[:3]
                keep.sort()  # preserve original channel order
                print(f"\t{path.name}: {C} channels, dropping ch {[c for c in range(C) if c not in keep]} (lowest avg)")
                image = image[keep]
            else:
                # pad with zero channels to reach 3
                print(f"\t{path.name}: {C} channels, padding with {3 - C} zero channel(s)")
                pad = np.zeros((3 - C, image.shape[1], image.shape[2]), dtype=image.dtype)
                image = np.concatenate([image, pad], axis=0)

        # CHW -> HWC for cv2
        image = np.transpose(image, (1, 2, 0))

        # resize preserving aspect ratio so longest side == _target
        H, W = image.shape[:2]
        scale = _target / max(H, W)
        new_W, new_H = int(W * scale), int(H * scale)
        image = cv2.resize(image, dsize=(new_W, new_H), interpolation=interpolation)

        # cv2 may squeeze single-channel to HW
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # zero-pad to _target x _target
        pad_H = _target - image.shape[0]
        pad_W = _target - image.shape[1]
        image = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), mode='constant', constant_values=0)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # normalize to float32 [0, 1]
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype != np.float32:
            image = image.astype(np.float32)
            vmin, vmax = image.min(), image.max()
            if vmax > vmin:
                image = (image - vmin) / (vmax - vmin)

        # Intended as a NO-OP: masks should already be binary here. Two of the
        # twenty source masks carry a non-zero background (4 and 5), which
        # reaches the loss as ~0.016 instead of 0, so the threshold always runs
        # and warns whenever it actually changed something.
        if is_mask:
            binary = (image > 0.5).astype(np.float32)
            if not np.array_equal(binary, image):
                print(f"\tWARNING {Path(path).name}: mask was not binary "
                      f"{np.unique(image)[:6]} -- thresholded at >0.5")
            image = binary

        return image    # (C, H, W) float32

    def collapse_channels(self, image):
        '''Collapse a (C, H, W) float32 array to (1, H, W) by averaging across channels.'''
        if image.shape[0] == 1:
            return image
        return image.mean(axis=0, keepdims=True)

    def gain_normalize(self, image, pct=99.9):
        '''
        Scale a (C, H, W) float32 array so its `pct` percentile reaches 1.0, clipped.

        Dividing by the dtype's full scale leaves BETWEEN-SLIDE exposure in the
        input: the 99.9th percentile runs from 0.07 to 0.70 across these twenty
        sections, a ~10x spread uncorrelated with whether the source was 8- or
        16-bit. BatchNorm does not absorb it, because it pools statistics over the
        BATCH rather than the sample -- with shuffle a batch mixes tiles from many
        slides and one affine is applied to all of them, so a dim slide's tile
        stays dim relative to a bright one's, and at inference the running
        statistics are frozen.

        ONE scalar for the whole array, never per channel: a per-channel factor
        would equalise the real MYL4-to-counterstain imbalance that the density
        quantification reports as a finding.

        A percentile rather than the maximum: a handful of saturated pixels
        otherwise set the factor and the section stays dark. Clipping costs
        0.0-0.4% of tissue pixels (measured over 529 tiles spanning all twenty
        slides), because the bright tail is rare in the ROI-adjacent region.

        Zeros are excluded -- the unscanned canvas and the bottom-right pad are
        zero, and including them drags the percentile toward the padding.

        Returns (scaled image, factor). Adopted on the basis of a two-fold pilot
        (folds 3 and 5): slide-level IoU +0.008 and +0.028 against freshly rerun
        controls, every held-out section improving, larger where the factor was
        larger.
        '''
        scanned = image > 0
        if not scanned.any():
            return image, 1.0
        hi = float(np.percentile(image[scanned], pct))
        if hi <= 0.0:
            return image, 1.0
        return np.clip(image / hi, 0.0, 1.0), 1.0 / hi

    def _make_vis(self, arr):
        '''Build an HWC uint8 RGB preview from a CHW float32 array, clipped at 99th percentile.'''
        if arr.shape[0] >= 3:
            vis = np.transpose(arr[:3], (1, 2, 0))
        else:
            vis = np.repeat(np.transpose(arr[:1], (1, 2, 0)), 3, axis=2)
        vmin = vis.min()
        vmax = np.percentile(vis, 99)
        vis = np.clip((vis - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        return (vis * 255).astype(np.uint8)

    def save_global(self, image, stem, subdir, dry_run=False):
        '''
        Save a (C, H, W) float32 ndarray as global.pt inside data_dir/<subdir>/<stem>/.
        In dry_run mode: prints stats and returns a HWC uint8 vis array instead of saving.

        Args:
            image (np.ndarray): CHW float32
            stem (str): image stem (used as subfolder name)
            subdir (str): 'images' or 'masks'
            dry_run (bool): if True, print stats and return vis array without saving
        '''
        tensor = torch.from_numpy(image)

        if dry_run:
            print(f"  [dry-run] {subdir}/{stem}/global.pt  shape={tuple(tensor.shape)}  "
                  f"min={tensor.min():.4f}  max={tensor.max():.4f}  mean={tensor.mean():.4f}")
            return self._make_vis(tensor.numpy())

        out_dir = Path(self.c['data_loader']['args']['data_dir']) / subdir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, out_dir / 'global.pt')

    def save_local(self, image, stem, subdir, dry_run=False):
        '''
        Extract patches from a CHW float32 image (already loaded at tiling_size
        resolution) and save each as a .pt file.

        Image and mask tiles are both saved at tile_size. The stride loop only
        visits fully in-bounds top-left positions, so no padding is needed.

        Patches saved to:  data_dir/<subdir>/<stem>/patch_YYYY_XXXX.pt
        Coordinates saved: data_dir/<subdir>/<stem>/coordinates.json
        '''
        tile_size = self.c['preprocess']['tile_size']
        stride    = self.c['preprocess']['stride']
        data_dir  = Path(self.c['data_loader']['args']['data_dir'])
        out_dir   = data_dir / subdir / stem

        img_hwc = np.transpose(image, (1, 2, 0))   # HWC for slicing
        H, W    = img_hwc.shape[:2]

        patch_info = []
        sample_patches = []

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                patch_hwc = img_hwc[y:y + tile_size, x:x + tile_size]

                # .copy() because the slice above is a view into img_hwc, and
                # torch.from_numpy would otherwise keep the whole canvas alive.
                patch = np.transpose(patch_hwc, (2, 0, 1)).copy()   # CHW
                tensor = torch.from_numpy(patch)

                filename = f'patch_{y:04d}_{x:04d}.pt'

                if not dry_run:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(tensor, out_dir / filename)
                elif len(sample_patches) < 9:
                    sample_patches.append(self._make_vis(patch))

                patch_info.append({
                    'filename': filename,
                    'stem': stem,
                    'x': x, 'y': y,
                    'width': tile_size, 'height': tile_size,
                    'image_W': W, 'image_H': H,
                })

        if not dry_run and patch_info:
            with open(out_dir / 'coordinates.json', 'w') as f:
                json.dump(patch_info, f, indent=2)

        n = len(patch_info)
        print(f"  [local] {subdir}/{stem}/patch_*  {n} patches  "
              f"(tile={tile_size}, stride={stride})  "
              f"({(H - tile_size) // stride + 1}y × {(W - tile_size) // stride + 1}x)")

        if dry_run and sample_patches:
            cols = 3
            rows = 3
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).flatten()
            for ax, vis in zip(axes, sample_patches):
                ax.imshow(vis)
                ax.axis('off')
            for ax in axes[len(sample_patches):]:
                ax.axis('off')
            fig.suptitle(f"local patches: {subdir}/{stem} (first {len(sample_patches)} of {n} total)")
            plt.tight_layout()
            plt.show()

        return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess images into global.pt + local patch_*.pt tensors per image")
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json)')
    parser.add_argument('--namescsv', default=None, type=str,
                        help='optional CSV with two whitespace-separated columns: filename and scene_index. '
                             'Only files whose stem matches are processed. Scene index selects which CZI scene '
                             'to load (0 by default). Duplicate stems with different scenes are both processed.')
    parser.add_argument('--nomask', action='store_true',
                        help='process images only, skip mask processing')
    parser.add_argument('--dry-run', action='store_true',
                        help='print stats and show previews without saving any files')
    args = parser.parse_args()

    # load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = json.load(f)

    # set up logging
    tile_dir = Path(config['data_loader']['args']['data_dir'])
    tile_dir.mkdir(exist_ok=True, parents=True)
    log_config = Path(__file__).parent / 'logger' / 'logger_config.json'
    setup_logging(tile_dir, log_config=str(log_config))
    logger = logging.getLogger('preprocess')

    # name filter + scene map from CSV
    names = None
    scene_map = {}   # stem -> scene_idx (last wins on duplicate stems)
    if args.namescsv is not None:
        with open(args.namescsv) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                stem = Path(parts[0]).stem
                scene_idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                if stem in scene_map and scene_map[stem] != scene_idx:
                    logger.warning(
                        f"duplicate stem '{stem}' in CSV with differing scene "
                        f"({scene_map[stem]} -> {scene_idx}); last entry wins"
                    )
                scene_map[stem] = scene_idx
        names = set(scene_map.keys())
        logger.info(f"name filter active: {len(names)} entries from {args.namescsv}")

    p = Preprocessor(config)

    image_paths, mask_paths = p.find_source_filepaths(names=names, nomask=args.nomask)

    if not args.dry_run:
        p.clear_existing_data()

    # Off by default, so an existing config reproduces the previous tensors byte
    # for byte. The factors are written beside them as gains.json: the number is
    # part of how the input was built, and a run should not have to be re-derived
    # to find out what it was.
    gain_normalize = config['preprocess'].get('gain_normalize', False)
    gains = {}
    if gain_normalize:
        logger.info("gain normalization ON: local tiles scaled so p99.9 -> 1.0")

    for i, image_path in enumerate(image_paths):
        clean_stem = image_path.stem.replace('_image', '').replace('.ome', '')
        scene_idx = scene_map.get(clean_stem, 0)
        logger.info(f"[{i+1}/{len(image_paths)}] {image_path.name} (scene {scene_idx})")

        mask_path = mask_paths[i] if not args.nomask else None
        if mask_path is not None:
            mask_stem = mask_path.stem.replace('_mask', '').replace('.ome', '')
            if mask_stem != clean_stem:
                logger.warning(f"[MISMATCH] image stem '{clean_stem}' != mask stem '{mask_stem}'")

        # --- global ---
        logger.info(f"[WORKING]\tglobal image...")
        collapse = config['preprocess'].get('collapse', False)
        image = p.load_and_format(image_path, cv2.INTER_LINEAR, is_mask=False,
                                  scene_index=scene_idx)
        if collapse:
            image = p.collapse_channels(image)
        image_vis = p.save_global(image, clean_stem, 'images', dry_run=args.dry_run)
        logger.info(f"[COMPLETE]\tglobal image  shape={image.shape}")

        mask_vis = None
        if mask_path is not None:
            logger.info(f"[WORKING]\tglobal mask...")
            mask = p.load_and_format(mask_path, cv2.INTER_NEAREST, is_mask=True,
                                     scene_index=scene_idx)
            mask_vis = p.save_global(mask, clean_stem, 'masks', dry_run=args.dry_run)
            logger.info(f"[COMPLETE]\tglobal mask  shape={mask.shape}")

        if args.dry_run:
            n_cols = 2 if mask_vis is not None else 1
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
            if n_cols == 1:
                axes = [axes]
            axes[0].imshow(image_vis)
            axes[0].set_title(f"global image: {clean_stem}")
            axes[0].axis('off')
            if mask_vis is not None:
                axes[1].imshow(mask_vis)
                axes[1].set_title(f"global mask: {clean_stem}")
                axes[1].axis('off')
            plt.tight_layout()
            plt.show()

        # --- local ---
        tiling_size = config['preprocess']['tiling_size']

        logger.info(f"[WORKING]\tlocal image tiles...")
        image_local = p.load_and_format(image_path, cv2.INTER_LINEAR, is_mask=False,
                                        target_size=tiling_size, scene_index=scene_idx)
        if collapse:
            image_local = p.collapse_channels(image_local)
        # Local tiles only. global.pt is annotation and preview material and is
        # never read during training (see LocalGlobalDataset's docstring), so
        # gaining it would widen the change without reaching the model.
        if gain_normalize:
            image_local, gain = p.gain_normalize(image_local)
            gains[clean_stem] = gain
            logger.info(f"[GAIN]\t\t{clean_stem}  p99.9 -> 1.0, factor {gain:.2f}x")
        n = p.save_local(image_local, clean_stem, 'images', dry_run=args.dry_run)
        logger.info(f"[COMPLETE]\tlocal image  {n} patches")

        if mask_path is not None:
            logger.info(f"[WORKING]\tlocal mask tiles...")
            mask_local = p.load_and_format(mask_path, cv2.INTER_NEAREST, is_mask=True,
                                           target_size=tiling_size, scene_index=scene_idx)
            n = p.save_local(mask_local, clean_stem, 'masks', dry_run=args.dry_run)
            logger.info(f"[COMPLETE]\tlocal mask  {n} patches")

        logger.info("")

    if gains and not args.dry_run:
        out = tile_dir / 'gains.json'
        out.write_text(json.dumps({'percentile': 99.9, 'source': 'local tiling canvas',
                                   'slides': gains}, indent=2))
        v = sorted(gains.values())
        logger.info(f"wrote {out}  ({len(gains)} slides, "
                    f"{v[0]:.2f}x - {v[-1]:.2f}x)")

    logger.info("preprocess complete")
