"""
Copy roi_<stem>.pt files from one preprocessed data directory to another.

Only copies ROIs for stems that exist in the destination directory.
ROIs are resolution-agnostic (stored relative to the global image), so they
transfer freely between datasets with different tiling_size or tile_size.

Usage:
    python utils/copy_rois.py --src data/[collapsed]ctx512_inner256_64tiles \
                               --dst data/[collapsed]t8192_ctx512_inner512
"""
import argparse
import shutil
from pathlib import Path


def copy_rois(src_dir, dst_dir, overwrite=False):
    src_images = Path(src_dir) / 'images'
    dst_images = Path(dst_dir) / 'images'

    roi_files = sorted(src_images.glob('*/roi_*.pt'))
    if not roi_files:
        print(f'No roi_*.pt files found under {src_images}')
        return

    copied = skipped = missing = 0
    for src_path in roi_files:
        stem = src_path.parent.name
        dst_stem_dir = dst_images / stem
        if not dst_stem_dir.exists():
            missing += 1
            continue
        dst_path = dst_stem_dir / src_path.name
        if dst_path.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f'copied={copied}  skipped={skipped}  missing_in_dst={missing}  total={len(roi_files)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='source data directory')
    parser.add_argument('--dst', required=True, help='destination data directory')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing ROIs in destination')
    args = parser.parse_args()
    copy_rois(args.src, args.dst, overwrite=args.overwrite)
