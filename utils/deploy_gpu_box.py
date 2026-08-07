"""Deploy the current branch to gpu-box by git, and report what landed.

Run this on workstation AFTER pushing. It only fetches and checks out on the remote;
it never rsyncs, so what runs on gpu-box is always a named commit that exists in
origin -- no "which copy is newer" ambiguity.

The gpu-box checkout lives at /mnt/1TBssd/mayo_kfold/att_vent_seg. data/ and
saved/ deliberately sit OUTSIDE it (in /mnt/1TBssd/mayo_kfold/) so a hard
checkout cannot touch them.

Usage:
    git push -u origin <branch>
    python utils/deploy_gpu_box.py                 # deploys current branch
    python utils/deploy_gpu_box.py --branch <b>
    python utils/deploy_gpu_box.py --check         # report remote state only
"""
import argparse
import subprocess
import sys

HOST = 'gpu-box@192.168.1.58'
REPO = '/mnt/1TBssd/mayo_kfold/att_vent_seg'


def local(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def remote(script, stdout_only=False):
    """stdout_only: git writes progress ("Reset branch ...") to stderr, which
    interleaves unpredictably with stdout -- so anything being parsed, rather
    than displayed, must read stdout alone."""
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST,
                        f'cd {REPO} && {script}'], capture_output=True, text=True)
    out = r.stdout.strip() if stdout_only else (r.stdout + r.stderr).strip()
    return r.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', default=None)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    branch = a.branch or local('git', 'rev-parse', '--abbrev-ref', 'HEAD')

    if a.check:
        _, out = remote('git rev-parse --abbrev-ref HEAD && git log --oneline -1 '
                        '&& git status --short')
        print(f"gpu-box {REPO}:\n{out}")
        return

    local_head = local('git', 'rev-parse', branch)
    unpushed = local('git', 'log', '--oneline', f'origin/{branch}..{branch}')
    if unpushed:
        sys.exit(f"refusing to deploy: {branch} has unpushed commits\n{unpushed}\n"
                 f"run: git push -u origin {branch}")

    rc, out = remote(f'git fetch --depth 1 origin {branch} '
                     f'&& git checkout -f -B {branch} origin/{branch}')
    print(out)
    if rc != 0:
        sys.exit(f"deploy failed (rc={rc})")

    # separate call, stdout only, so no progress chatter can be misread as a sha
    _, remote_head = remote('git rev-parse HEAD', stdout_only=True)
    _, desc = remote('git log --oneline -1', stdout_only=True)
    if remote_head != local_head:
        sys.exit(f"MISMATCH: local {local_head[:8]} != gpu-box {remote_head[:8]}")
    print(f"\nOK: gpu-box at {remote_head[:8]} on {branch} (matches local)\n    {desc}")


if __name__ == '__main__':
    main()
