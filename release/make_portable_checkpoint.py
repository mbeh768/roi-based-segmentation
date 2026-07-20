"""
Convert a training checkpoint into a portable, inference-only one.

Training checkpoints pickle the live `ConfigParser` object, which holds
`pathlib.Path` attributes. A Path created on Linux pickles as `PosixPath`, and
unpickling that on Windows raises:

    NotImplementedError: cannot instantiate 'PosixPath' on your system

so the weights simply cannot be loaded on a collaborator's machine. Rebuilding
the checkpoint out of plain data (dicts, strings, tensors) fixes that
permanently and has two further benefits:

  * it loads with `weights_only=True`, so no arbitrary Python classes are ever
    reconstructed -- which is both safer and the reason it is now portable;
  * it needs none of this repo's modules on the import path, so the file works
    for anyone who downloads it from Hugging Face.

The optimizer state is dropped: it is only useful for resuming training, and it
is a large fraction of the file.

Usage:
    python make_portable_checkpoint.py <in.pth> <out.pth>
"""
import argparse
import pathlib
import sys

import torch


def to_plain(obj, depth=0):
    """Recursively convert config data to JSON-ish primitives."""
    if depth > 12:
        return obj
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v, depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Anything else (a ConfigParser, a custom object) -- unwrap its dict form.
    if hasattr(obj, 'config'):
        return to_plain(obj.config, depth + 1)
    if hasattr(obj, '__dict__'):
        return to_plain(vars(obj), depth + 1)
    return str(obj)


def find_non_portable(obj, path='ckpt', depth=0, hits=None):
    """Report anything that is not a primitive/dict/list/tensor."""
    if hits is None:
        hits = []
    if depth > 12:
        return hits
    if isinstance(obj, pathlib.PurePath):
        hits.append((path, type(obj).__name__))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_non_portable(v, f'{path}[{k!r}]', depth + 1, hits)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_non_portable(v, f'{path}[{i}]', depth + 1, hits)
    elif isinstance(obj, torch.Tensor):
        pass
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        pass
    else:
        hits.append((path, type(obj).__name__))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()

    # The source still contains a ConfigParser, so it needs this repo importable.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

    print(f'Reading {args.src} ...')
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)

    config = to_plain(ckpt['config'])

    # The encoder's ImageNet weights are downloaded from the internet when
    # backbone_pretrained is true. For inference that download is both a failure
    # point on an offline machine and pointless, because load_state_dict
    # immediately overwrites every one of those weights with the trained ones.
    arch_args = config.get('arch', {}).get('args', {})
    if arch_args.get('backbone_pretrained'):
        arch_args['backbone_pretrained'] = False
        print('  set arch.args.backbone_pretrained = false (no download needed)')

    portable = {
        'arch': str(ckpt.get('arch', config.get('arch', {}).get('type', 'unknown'))),
        'epoch': int(ckpt.get('epoch', -1)),
        'state_dict': {k: v for k, v in ckpt['state_dict'].items()},
        'config': config,
    }

    bad = find_non_portable(portable)
    if bad:
        print('\nERROR: non-portable objects remain:')
        for p, t in bad:
            print(f'  {p}: {t}')
        return 1

    torch.save(portable, args.dst)

    # Prove it: weights_only=True refuses to reconstruct arbitrary classes, so a
    # successful load here is a guarantee that no PosixPath (or anything else
    # platform specific) can surface on another machine.
    reloaded = torch.load(args.dst, map_location='cpu', weights_only=True)
    assert reloaded['epoch'] == portable['epoch']

    src_mb = pathlib.Path(args.src).stat().st_size / 1e6
    dst_mb = pathlib.Path(args.dst).stat().st_size / 1e6
    print(f'\nWrote {args.dst}')
    print(f'  epoch      : {portable["epoch"]}')
    print(f'  arch       : {portable["arch"]}')
    print(f'  size       : {src_mb:.1f} MB -> {dst_mb:.1f} MB')
    print(f'  weights_only=True load: OK (portable, no pickled classes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
