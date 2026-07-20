"""
Dependency installer for the ventricle segmentation tool.

Run by run_windows.bat / run_mac_linux.sh inside the freshly created .venv.
Uses only the Python standard library, because it runs *before* anything is
installed.

The one interesting job here is picking a PyTorch build:

  * PyTorch's CUDA wheels ship their own CUDA runtime, so a collaborator does
    NOT need to install the CUDA toolkit. All they need is an NVIDIA driver.
  * `nvidia-smi` reports the highest CUDA version that driver supports. We pick
    the newest PyTorch CUDA channel at or below that number.
  * If anything about that guess is wrong, the install is verified afterwards by
    actually importing torch and asking it whether CUDA works. If it doesn't, we
    reinstall the CPU build automatically. That means a bad guess costs time, not
    a broken install.

Usage:
    python install_deps.py [--auto|--gpu|--cpu] [--yes]
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Newest first. Each entry is (driver CUDA version required, wheel channel).
# Only channels PyTorch actually publishes belong here.
CUDA_CHANNELS = [
    ((13, 2), 'cu132'),
    ((13, 0), 'cu130'),
    ((12, 9), 'cu129'),
    ((12, 8), 'cu128'),
    ((12, 6), 'cu126'),
    ((12, 4), 'cu124'),
    ((12, 1), 'cu121'),
    ((11, 8), 'cu118'),
]

CPU_INDEX = 'https://download.pytorch.org/whl/cpu'

# torch/torchvision are installed separately from a CPU or CUDA specific index,
# so they are excluded here.
_TORCH_PACKAGES = {'torch', 'torchvision'}


def other_packages():
    """Everything except torch, read from requirements.txt.

    Read from the file rather than duplicated in a list here, so the two can
    never drift apart.
    """
    req = Path(__file__).resolve().parent / 'requirements.txt'
    packages = []
    for line in req.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name = re.split(r'[<>=!\[]', line, 1)[0].strip().lower()
        if name not in _TORCH_PACKAGES:
            packages.append(line)
    return packages


def run(cmd, **kw):
    print('  $', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def detect_gpu():
    """Return (gpu_name, (major, minor)) from nvidia-smi, or (None, None)."""
    if not shutil.which('nvidia-smi'):
        return None, None
    try:
        out = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                             timeout=30)
    except Exception:
        return None, None
    if out.returncode != 0:
        return None, None
    text = out.stdout

    m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', text)
    cuda = (int(m.group(1)), int(m.group(2))) if m else None

    name = None
    # GPU model sits in the table rows, e.g. "|   0  NVIDIA GeForce RTX 4070 ..."
    n = re.search(r'\|\s*\d+\s+(NVIDIA[^|]*?)\s{2,}', text)
    if n:
        name = n.group(1).strip()
    return name, cuda


def pick_channel(driver_cuda):
    """Newest published CUDA channel the driver can run, or None."""
    if not driver_cuda:
        return None
    for required, channel in CUDA_CHANNELS:
        if driver_cuda >= required:
            return channel
    return None


def pip_install(py, args):
    return run([py, '-m', 'pip', 'install'] + args).returncode == 0


def install_torch(py, index_url, label):
    """Install torch/torchvision from a specific index, replacing any existing build.

    The CPU and CUDA wheels share a version number (e.g. 2.13.0 either way, the
    build differing only in local metadata), so a plain `pip install` decides the
    requirement is already satisfied and silently leaves the old build in place.
    That made switching between CPU and GPU a no-op that still reported success.
    Uninstalling first forces the requested build to actually land. On a fresh
    environment there is nothing to remove, so this costs nothing.
    """
    print(f'\nInstalling PyTorch ({label}) ...')
    subprocess.run([py, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision'],
                   capture_output=True, text=True)
    return pip_install(py, ['torch', 'torchvision', '--index-url', index_url])


def cuda_works(py):
    """Import torch in a subprocess and ask whether CUDA is actually usable."""
    probe = (
        'import torch;'
        'ok = torch.cuda.is_available();'
        'print("CUDA_OK" if ok else "CUDA_MISSING");'
        'print(torch.cuda.get_device_name(0) if ok else "")'
    )
    out = subprocess.run([py, '-c', probe], capture_output=True, text=True)
    if 'CUDA_OK' in out.stdout:
        name = out.stdout.strip().splitlines()[-1].strip()
        return True, name
    return False, out.stdout.strip() + out.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--auto', action='store_true', default=True,
                      help='detect a GPU and ask (default)')
    mode.add_argument('--gpu', action='store_true', help='force the CUDA build')
    mode.add_argument('--cpu', action='store_true', help='force the CPU build')
    ap.add_argument('--yes', action='store_true',
                    help='do not prompt; take the detected default')
    args = ap.parse_args()

    py = sys.executable
    print('Upgrading pip ...')
    pip_install(py, ['--upgrade', 'pip'])

    gpu_name, driver_cuda = detect_gpu()
    channel = pick_channel(driver_cuda)

    print('\n--- Hardware check ---')
    if gpu_name:
        print(f'  NVIDIA GPU     : {gpu_name}')
        print(f'  Driver supports: CUDA {driver_cuda[0]}.{driver_cuda[1]}'
              if driver_cuda else '  Driver supports: unknown')
        print(f'  Matching build : {channel or "none (driver too old)"}')
    else:
        print('  No NVIDIA GPU detected.')

    want_gpu = False
    if args.gpu:
        want_gpu = True
        if not channel:
            # Respect the explicit request but pick something plausible, then
            # let the post-install verification catch it if it cannot work.
            channel = CUDA_CHANNELS[-1][1]
            print('  (forcing GPU build despite no usable detection)')
    elif args.cpu:
        want_gpu = False
    elif gpu_name and channel:
        if args.yes:
            want_gpu = True
        else:
            print('\nA GPU build will make segmentation several times faster,')
            print('but downloads roughly 3 GB instead of about 300 MB.')
            try:
                answer = input('Install the GPU version? [Y/n]: ').strip().lower()
            except EOFError:
                answer = ''
            want_gpu = answer in ('', 'y', 'yes')

    if want_gpu:
        index = f'https://download.pytorch.org/whl/{channel}'
        if not install_torch(py, index, f'GPU / {channel}'):
            print('\nGPU build failed to install; falling back to the CPU build.')
            want_gpu = False
            if not install_torch(py, CPU_INDEX, 'CPU'):
                return 1
    else:
        if not install_torch(py, CPU_INDEX, 'CPU'):
            return 1

    print('\nInstalling remaining packages ...')
    if not pip_install(py, other_packages()):
        return 1

    # Verify rather than assume. A CUDA wheel can install cleanly and still be
    # unusable (driver too old, GPU not visible in this session), so confirm and
    # self-correct instead of leaving a broken environment behind.
    if want_gpu:
        print('\nVerifying GPU support ...')
        ok, detail = cuda_works(py)
        if ok:
            print(f'  GPU is working: {detail}')
        else:
            print('  GPU build installed but CUDA is not usable here.')
            print(f'  ({detail.splitlines()[0] if detail else "no detail"})')
            print('  Reinstalling the CPU build so the tool still works ...')
            if not install_torch(py, CPU_INDEX, 'CPU'):
                return 1
            want_gpu = False

    print('\n' + '=' * 55)
    print(' Setup complete. Using the '
          + ('GPU' if want_gpu else 'CPU') + ' build of PyTorch.')
    print('=' * 55)
    return 0


if __name__ == '__main__':
    sys.exit(main())
