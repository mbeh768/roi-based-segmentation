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
            image = image[np.newaxis, :, :]   # HW -> CHW

        return image                           # CHW original dtype

    def _load_czi(self, path):
        '''Load CZI image as CHW array, zoomed so pixel count doesn't exceed czi_max_px.

        Requires: pylibCZIrw  (pip install pylibCZIrw)
        '''
        from pylibCZIrw import czi as pyczi

        _max_czi_px = self.c['preprocess']['czi_max_px']
        with pyczi.open_czi(str(path)) as czidoc:
            bb = czidoc.total_bounding_box
            H = bb['Y'][1] - bb['Y'][0]
            W = bb['X'][1] - bb['X'][0]
            zoom = 1.0
            while zoom > 0.0 and (H * zoom) * (W * zoom) > _max_czi_px:
                zoom *= 0.5
            print(f"\tczi image dimensions: {H}x{W} ({H*W:.2e} px), pyramid zoom: {zoom} -> {int(H*zoom)}x{int(W*zoom)} ({H*zoom*W*zoom:.2e} px)\n")
            scenes = list(czidoc.scenes_bounding_rectangle.keys())
            scene = scenes[0] if scenes else None
            data = czidoc.read(scene=scene, zoom=zoom)    # (H, W, C)

        data = np.transpose(data, (2, 0, 1))              # CHW

        return data                                        # CHW original dtype

    def load_and_format(self, path, interpolation, standardize=True, is_mask=False, target_size=None, imagenet_norm=False):
        '''
        Load an image or mask, resize to a square, normalize to float32 [0, 1],
        optionally standardize channels per-channel, zero-pad to square.

        Args:
            path (str|Path): path to image or mask file
            interpolation: cv2 interpolation flag for resizing
            standardize (bool): if True, apply per-channel min-max stretching
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
            image = self._load_czi(path)
        elif is_mask:
            image = self._load_tif_tifffile(path)
        else:
            try:
                image = self._load_tif_openslide(path, _target)
            except Exception as e:
                print(f"\topenslide failed ({type(e).__name__}), falling back to tifffile")
                image = self._load_tif_tifffile(path)

        # enforce 3-channel images (CHW at this point)
        if not is_mask and image.shape[0] != 3:
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

        if standardize:
            for c in range(image.shape[0]):
                ch_min = image[c].min()
                ch_max = image[c].max()
                if ch_max > ch_min:
                    image[c] = (image[c] - ch_min) / (ch_max - ch_min)

        if imagenet_norm and not is_mask:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            image = (image - mean) / std

        return image    # (C, H, W) float32

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
        Extract tile_size x tile_size patches from a CHW float32 image (already loaded
        at tiling_size resolution) and save each as a .pt file.

        Patches saved to:  data_dir/<subdir>/local/<stem>/patch_YYYY_XXXX.pt
        Coordinates saved: data_dir/<subdir>/local/<stem>/coordinates.json

        Args:
            image (np.ndarray): CHW float32, spatial dims should match tiling_size
            stem (str): source image stem (used as subfolder name)
            subdir (str): 'images' or 'masks'
            dry_run (bool): if True, print stats and show a sample grid without saving

        Returns:
            int: number of patches extracted
        '''
        tile_size = self.c['preprocess']['tile_size']
        stride    = self.c['preprocess']['stride']
        data_dir  = Path(self.c['data_loader']['args']['data_dir'])
        out_dir   = data_dir / subdir / stem

        img_hwc = np.transpose(image, (1, 2, 0))   # HWC for slicing
        H, W    = img_hwc.shape[:2]

        patch_info = []
        sample_patches = []   # kept for dry-run preview

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                patch = np.transpose(
                    img_hwc[y:y+tile_size, x:x+tile_size], (2, 0, 1)   # CHW
                ).copy()
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
                        help='optional CSV with a "name" column; only files whose stem matches will be processed')
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

    # name filter from CSV
    names = None
    if args.namescsv is not None:
        with open(args.namescsv) as f:
            names = {Path(line.strip()).stem for line in f if line.strip()}
        logger.info(f"name filter active: {len(names)} entries from {args.namescsv}")

    p = Preprocessor(config)

    image_paths, mask_paths = p.find_source_filepaths(names=names, nomask=args.nomask)

    if not args.dry_run:
        p.clear_existing_data()

    for i, image_path in enumerate(image_paths):
        clean_stem = image_path.stem.replace('_image', '').replace('.ome', '')
        logger.info(f"[{i+1}/{len(image_paths)}] {image_path.name}")

        mask_path = mask_paths[i] if not args.nomask else None
        if mask_path is not None:
            mask_stem = mask_path.stem.replace('_mask', '').replace('.ome', '')
            if mask_stem != clean_stem:
                logger.warning(f"[MISMATCH] image stem '{clean_stem}' != mask stem '{mask_stem}'")

        # --- global ---
        logger.info(f"[WORKING]\tglobal image...")
        imagenet_norm = config['preprocess'].get('imagenet_norm', False)
        image = p.load_and_format(image_path, cv2.INTER_LINEAR, is_mask=False, imagenet_norm=imagenet_norm)
        image_vis = p.save_global(image, clean_stem, 'images', dry_run=args.dry_run)
        logger.info(f"[COMPLETE]\tglobal image  shape={image.shape}")

        mask_vis = None
        if mask_path is not None:
            logger.info(f"[WORKING]\tglobal mask...")
            mask = p.load_and_format(mask_path, cv2.INTER_NEAREST, standardize=False, is_mask=True)
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
                                        target_size=tiling_size, imagenet_norm=imagenet_norm)
        n = p.save_local(image_local, clean_stem, 'images', dry_run=args.dry_run)
        logger.info(f"[COMPLETE]\tlocal image  {n} patches")

        if mask_path is not None:
            logger.info(f"[WORKING]\tlocal mask tiles...")
            mask_local = p.load_and_format(mask_path, cv2.INTER_NEAREST,
                                           standardize=False, is_mask=True,
                                           target_size=tiling_size)
            n = p.save_local(mask_local, clean_stem, 'masks', dry_run=args.dry_run)
            logger.info(f"[COMPLETE]\tlocal mask  {n} patches")

        logger.info("")

    logger.info("preprocess complete")
