"""
Validate that SSL checkpoint weights transfer correctly into
PretrainedLocalGlobalUNet.global_enc.

Usage:
    python utils/validate_ssl_transfer.py <ssl_checkpoint_path>
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.model import PretrainedLocalGlobalUNet


def validate(ssl_path):
    # --- load raw SSL state dict ---
    ckpt = torch.load(ssl_path, map_location='cpu', weights_only=False)
    ssl_sd = ckpt['state_dict']
    expected = {k.replace('unet.', ''): v
                for k, v in ssl_sd.items() if k.startswith('unet.')}

    # --- infer in_ch from the SSL checkpoint's first conv weight ---
    in_ch = expected['enc1.0.weight'].shape[1]

    # --- build model with that checkpoint ---
    model = PretrainedLocalGlobalUNet(ssl_checkpoint=ssl_path, in_ch=in_ch)
    actual = model.global_enc.state_dict()

    # 1) Key check
    missing = set(expected) - set(actual)
    extra   = set(actual)   - set(expected)
    ok = True
    if missing:
        print(f"FAIL: keys in SSL but not in global_enc: {missing}")
        ok = False
    if extra:
        print(f"FAIL: keys in global_enc but not in SSL: {extra}")
        ok = False
    if not missing and not extra:
        print(f"OK: all {len(expected)} keys match")

    # 2) Value check
    mismatched = []
    for k in sorted(set(expected) & set(actual)):
        if not torch.equal(expected[k], actual[k]):
            mismatched.append(k)
    if mismatched:
        print(f"FAIL: {len(mismatched)} tensors differ after load:")
        for k in mismatched:
            print(f"  {k}")
        ok = False
    else:
        print(f"OK: all tensor values match")

    # 3) Grad check — report which params have requires_grad
    print("\nParameter grad status in global_enc:")
    for name, p in model.global_enc.named_parameters():
        print(f"  {'TRAIN' if p.requires_grad else 'FROZEN'}  {name}  {tuple(p.shape)}")

    if ok:
        print("\nAll checks passed.")
    return ok


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python utils/validate_ssl_transfer.py <checkpoint.pth>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
