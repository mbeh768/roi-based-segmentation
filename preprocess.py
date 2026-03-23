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

    def _load_tif_openslide(self, path):
        '''Load TIF/TIFF via OpenSlide at the best pyramid level for the target size.

        Returns CHW uint8 ndarray (3 channels, RGB).
        Raises openslide.OpenSlideUnsupportedFormatError if the file is not supported.
        '''
        import openslide

        _max_dim = self.c['preprocess']['downsample']
        slide = openslide.OpenSlide(str(path))
        W, H = slide.dimensions
        downsample_factor = max(W, H) / _max_dim
        level = slide.get_best_level_for_downsample(downsample_factor)
        lw, lh = slide.level_dimensions[level]

        print(f"\topenslide: {W}x{H} -> reading level {level} ({lw}x{lh})")

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

    def load_and_format(self, path, interpolation, standardize=True, is_mask=False):
        '''
        Load an image or mask, resize to a square (downsample x downsample), normalize
        to float32 [0, 1], optionally standardize channels per-channel.

        Args:
            path (str|Path): path to image or mask file
            interpolation: cv2 interpolation flag for resizing
            standardize (bool): if True, apply per-channel min-max stretching
            is_mask (bool): if True, use tifffile loader directly (preserves label values)

        Returns:
            np.ndarray: shape (C, H, W) float32, where C=3 for RGB images, C=1 for masks
        '''
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == '.czi':
            image = self._load_czi(path)
        elif is_mask:
            image = self._load_tif_tifffile(path)
        else:
            try:
                import openslide
                image = self._load_tif_openslide(path)
            except Exception as e:
                print(f"\topenslide failed ({type(e).__name__}), falling back to tifffile")
                image = self._load_tif_tifffile(path)

        _target = self.c['preprocess']['downsample']

        # CHW -> HWC for cv2
        image = np.transpose(image, (1, 2, 0))
        H, W = image.shape[:2]
        scale = _target / max(H, W)
        new_W, new_H = int(W * scale), int(H * scale)
        image = cv2.resize(image, dsize=(new_W, new_H), interpolation=interpolation)

        # cv2 may squeeze single-channel to HW
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

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
        Save a (C, H, W) float32 ndarray as a .pt tensor.
        In dry_run mode: prints stats and returns a HWC uint8 vis array instead of saving.

        Args:
            image (np.ndarray): CHW float32
            stem (str): output filename stem
            subdir (str): 'images' or 'masks'
            dry_run (bool): if True, print stats and return vis array without saving
        '''
        tensor = torch.from_numpy(image)

        if dry_run:
            print(f"  [dry-run] {subdir}/{stem}.pt  shape={tuple(tensor.shape)}  "
                  f"min={tensor.min():.4f}  max={tensor.max():.4f}  mean={tensor.mean():.4f}")
            return self._make_vis(tensor.numpy())

        out_dir = Path(self.c['data_loader']['args']['data_dir']) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, out_dir / f'{stem}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess images into downsampled global .pt tensors")
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

        logger.info(f"[WORKING]\tloading image...")
        image = p.load_and_format(image_path, cv2.INTER_LINEAR, is_mask=False)
        logger.info(f"[WORKING]\tsaving global image  shape={image.shape}")
        image_vis = p.save_global(image, clean_stem, 'images', dry_run=args.dry_run)
        logger.info(f"[COMPLETE]\timage")

        mask_vis = None
        if not args.nomask:
            mask_path = mask_paths[i]
            mask_stem = mask_path.stem.replace('_mask', '').replace('.ome', '')
            if mask_stem != clean_stem:
                logger.warning(f"[MISMATCH] image stem '{clean_stem}' != mask stem '{mask_stem}'")
            logger.info(f"[WORKING]\tloading mask {mask_path.name}...")
            mask = p.load_and_format(mask_path, cv2.INTER_NEAREST, standardize=False, is_mask=True)
            logger.info(f"[WORKING]\tsaving global mask  shape={mask.shape}")
            mask_vis = p.save_global(mask, clean_stem, 'masks', dry_run=args.dry_run)
            logger.info(f"[COMPLETE]\tmask")

        if args.dry_run:
            n_cols = 2 if mask_vis is not None else 1
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
            if n_cols == 1:
                axes = [axes]
            axes[0].imshow(image_vis)
            axes[0].set_title(f"image: {clean_stem}")
            axes[0].axis('off')
            if mask_vis is not None:
                axes[1].imshow(mask_vis)
                axes[1].set_title(f"mask: {clean_stem}")
                axes[1].axis('off')
            plt.tight_layout()
            plt.show()

        logger.info("")

    logger.info("preprocess complete")
