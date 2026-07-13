"""
Gathers live audit evidence for the held-out-model-vs-ground-truth verification
report: checksums of every input file actually used, timestamps proving
chronological ordering (checkpoint -> prediction -> quantification), and
train/val set membership lists pulled fresh from laptop over SSH.

Writes evidence/<pair>.json. Re-run any time to refresh/re-verify.

Usage:
    python verification/gather_evidence.py AJK056-10
    python verification/gather_evidence.py AJKY121-9
    python verification/gather_evidence.py AJKY121-8
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
RGBROOT = ("/mnt/8TBhdd/datasets/mayo_datasets/Max B_files_from Behfar_Shared "
           "folder(formatted)/Segmentation_Project/RGB FULL - dataset")
CZIROOT = ("/mnt/8TBhdd/datasets/mayo_datasets/Max B_files_from Behfar_Shared "
           "folder(formatted)/Segmentation_Project/Provided Dataset")

PAIRS = {
    "AJK056-10": {
        "run": "EXPERIMENT 10A_old [wholistic performance evaluation]_2",
        "stems": ["AJKV056-10-1-of-2", "AJK056-10-2of2"],
        "gt_mask": {
            "AJKV056-10-1-of-2": f"{RGBROOT}/AJK056-10-1of2/AJKV056-10-1-of-2_mask.tiff",
            "AJK056-10-2of2": f"{RGBROOT}/AJK056-10-2of2/AJK056-10-2of2_mask.tiff",
        },
        "raw_image": {
            "AJKV056-10-1-of-2": f"{RGBROOT}/AJK056-10-1of2/AJKV056-10-1-of-2_image.tiff",
            "AJK056-10-2of2": f"{RGBROOT}/AJK056-10-2of2/AJK056-10-2of2_image.tiff",
        },
        "gt_intensity_dir": {
            "AJKV056-10-1-of-2": "pairs_quant_gt/AJK056-10_pair/intensity_1of2",
            "AJK056-10-2of2": "pairs_quant_gt/AJK056-10_pair/intensity_2of2",
        },
        "pred_prob_map": {
            "AJKV056-10-1-of-2": "held_out_results/AJK056-10_pair/AJKV056-10-1-of-2_prob_map.tif",
            "AJK056-10-2of2": "held_out_results/AJK056-10_pair/AJK056-10-2of2_prob_map.tif",
        },
        "native_pred_mask": {
            "AJKV056-10-1-of-2": "/tmp/native_pred_masks/AJKV056-10-1-of-2_native_pred_mask.tif",
            "AJK056-10-2of2": "/tmp/native_pred_masks/AJK056-10-2of2_native_pred_mask.tif",
        },
        "pred_quant_dir": {
            "AJKV056-10-1-of-2": "held_out_results/AJK056-10_pair/quantification_pred",
            "AJK056-10-2of2": "held_out_results/AJK056-10_pair/quantification_2of2_pred",
        },
        "pred_intensity_dir": {
            "AJKV056-10-1-of-2": "held_out_results/AJK056-10_pair/intensity_1of2_pred",
            "AJK056-10-2of2": "held_out_results/AJK056-10_pair/intensity_2of2_pred",
        },
    },
    "AJKY121-9": {
        "run": "EXPERIMENT 10A_old [wholistic performance evaluation]_3",
        "stems": ["AJKY121-9-1", "AJKY121-9-2"],
        "gt_mask": {
            "AJKY121-9-1": f"{RGBROOT}/AJKY121_9/AJKY121-9-1_mask.tif",
            "AJKY121-9-2": f"{RGBROOT}/AJKY121_9/AJKY121-9-2_mask.tif",
        },
        "raw_image": {
            "AJKY121-9-1": f"{RGBROOT}/AJKY121_9/AJKY121-9-1_image.tif",
            "AJKY121-9-2": f"{RGBROOT}/AJKY121_9/AJKY121-9-2_image.tif",
        },
        "gt_intensity_dir": {
            "AJKY121-9-1": "pairs_quant_gt/AJKY121-9_pair/intensity_1of2",
            "AJKY121-9-2": "pairs_quant_gt/AJKY121-9_pair/intensity_2of2",
        },
        "pred_prob_map": {
            "AJKY121-9-1": "held_out_results/AJKY121-9_pair/AJKY121-9-1_prob_map.tif",
            "AJKY121-9-2": "held_out_results/AJKY121-9_pair/AJKY121-9-2_prob_map.tif",
        },
        "native_pred_mask": {
            "AJKY121-9-1": "/tmp/native_pred_masks/AJKY121-9-1_native_pred_mask.tif",
            "AJKY121-9-2": "/tmp/native_pred_masks/AJKY121-9-2_native_pred_mask.tif",
        },
        "pred_quant_dir": {
            "AJKY121-9-1": "held_out_results/AJKY121-9_pair/quantification_pred",
            "AJKY121-9-2": "held_out_results/AJKY121-9_pair/quantification_2of2_pred",
        },
        "pred_intensity_dir": {
            "AJKY121-9-1": "held_out_results/AJKY121-9_pair/intensity_1of2_pred",
            "AJKY121-9-2": "held_out_results/AJKY121-9_pair/intensity_2of2_pred",
        },
    },
    "AJKY121-8": {
        "run": "EXPERIMENT 10A_old [wholistic performance evaluation]_6",
        "stems": ["AJKY121-8-1", "AJKY121-8-2"],
        "gt_mask": {
            "AJKY121-8-1": f"{RGBROOT}/AJKY121_8/AJKY121-8-1_mask.tif",
            "AJKY121-8-2": f"{RGBROOT}/AJKY121_8/AJKY121-8-2_mask.tif",
        },
        "raw_image": {
            "AJKY121-8-1": f"{RGBROOT}/AJKY121_8/AJKY121-8-1_image.tif",
            "AJKY121-8-2": f"{RGBROOT}/AJKY121_8/AJKY121-8-2_image.tif",
        },
        "gt_intensity_dir": {
            "AJKY121-8-1": "pairs_quant_gt/AJKY121-8_pair/intensity_1of2",
            "AJKY121-8-2": "pairs_quant_gt/AJKY121-8_pair/intensity_2of2",
        },
        "pred_prob_map": {
            "AJKY121-8-1": "held_out_results/AJKY121-8_pair/AJKY121-8-1_prob_map.tif",
            "AJKY121-8-2": "held_out_results/AJKY121-8_pair/AJKY121-8-2_prob_map.tif",
        },
        "native_pred_mask": {
            "AJKY121-8-1": "/tmp/native_pred_masks/AJKY121-8-1_native_pred_mask.tif",
            "AJKY121-8-2": "/tmp/native_pred_masks/AJKY121-8-2_native_pred_mask.tif",
        },
        "pred_quant_dir": {
            "AJKY121-8-1": "held_out_results/AJKY121-8_pair/quantification_pred",
            "AJKY121-8-2": "held_out_results/AJKY121-8_pair/quantification_2of2_pred",
        },
        "pred_intensity_dir": {
            "AJKY121-8-1": "held_out_results/AJKY121-8_pair/intensity_1of2_pred",
            "AJKY121-8-2": "held_out_results/AJKY121-8_pair/intensity_2of2_pred",
        },
    },
}


def md5(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime(path):
    p = Path(path)
    if not p.exists():
        return None
    import datetime
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()


def remote_mtime(remote_relpath):
    """Best-effort: stat a file on laptop (path relative to /home/USER/scripts) and
    return its mtime as a unix float, or None if unreachable/missing.

    This is the AUTHORITATIVE chronology source. Local mtimes on workstation only reflect
    when a file happened to be *transferred* here -- which, whenever a transfer had
    to be retried after a crash (this session hit several), can end up completely
    scrambled relative to the true generation order on laptop. Comparing remote
    stat times sidesteps that entirely.
    """
    out = subprocess.run(
        f'ssh gpu-box \'wsl.exe -- bash -ic "stat -c %Y /home/USER/scripts/{remote_relpath} 2>/dev/null"\'',
        shell=True, capture_output=True, text=True, timeout=15,
    )
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    try:
        return float(line)
    except ValueError:
        return None


def ssh_laptop(remote_script_body):
    """Run a bash script body on laptop over the tar-pipe deploy pattern, return stdout."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        script_path = Path(td) / "audit.sh"
        script_path.write_text(remote_script_body)
        subprocess.run(
            f'tar czf - -C {td} audit.sh | ssh gpu-box '
            f'\'wsl.exe -- bash -ic "cd /home/laptop && tar xzf -"\'',
            shell=True, check=True, capture_output=True,
        )
        out = subprocess.run(
            'ssh gpu-box \'wsl.exe -- bash -ic "bash /home/USER/audit.sh"\'',
            shell=True, capture_output=True, text=True,
        )
        return out.stdout


def gather(pair_name):
    cfg = PAIRS[pair_name]
    run = cfg["run"]
    evidence = {"pair": pair_name, "run_dir": run, "stems": cfg["stems"]}

    # 1. train/val set membership, pulled live from laptop
    remote = (
        f'echo "TRAIN:"; ls "/home/USER/scripts/saved/log/{run}/sets/train" 2>&1\n'
        f'echo "VAL:"; ls "/home/USER/scripts/saved/log/{run}/sets/val" 2>&1\n'
        f'echo "CKPT_STAT:"; stat -c "%Y %n" "/home/USER/scripts/saved/models/{run}/model_best.pth" 2>&1\n'
    )
    out = ssh_laptop(remote)
    sections = {"TRAIN:": [], "VAL:": [], "CKPT_STAT:": []}
    cur = None
    for line in out.splitlines():
        if line.strip() in sections:
            cur = line.strip()
            continue
        if cur and line.strip():
            sections[cur].append(line.strip())
    train_stems = [s.replace(".png", "") for s in sections["TRAIN:"]]
    val_stems = [s.replace(".png", "") for s in sections["VAL:"]]
    evidence["train_set_stems"] = sorted(train_stems)
    evidence["val_set_stems"] = sorted(val_stems)
    evidence["held_out_stems_absent_from_train"] = [
        s for s in cfg["stems"] if s not in train_stems
    ]
    evidence["held_out_stems_absent_from_val"] = [
        s for s in cfg["stems"] if s not in val_stems
    ]
    evidence["exclusion_verified"] = (
        len(evidence["held_out_stems_absent_from_train"]) == len(cfg["stems"])
        and len(evidence["held_out_stems_absent_from_val"]) == len(cfg["stems"])
    )
    evidence["checkpoint_remote_stat"] = sections["CKPT_STAT:"]

    # 2. per-stem file chain: gt mask, raw image, prob_map, native pred mask, outputs
    evidence["stems_detail"] = {}
    for stem in cfg["stems"]:
        d = {}
        d["gt_mask_path"] = cfg["gt_mask"][stem]
        d["gt_mask_md5"] = md5(cfg["gt_mask"][stem])
        d["raw_image_path"] = cfg["raw_image"][stem]
        d["raw_image_md5"] = md5(cfg["raw_image"][stem])

        gt_dir = REPO / cfg["gt_intensity_dir"][stem]
        d["gt_radial_profile_md5"] = md5(gt_dir / "radial_profile.csv")
        d["gt_band_means_csv"] = (gt_dir / "band_means.csv").read_text() if (gt_dir / "band_means.csv").exists() else None

        prob_map = REPO / cfg["pred_prob_map"][stem]
        d["pred_prob_map_path"] = str(prob_map)
        d["pred_prob_map_md5"] = md5(prob_map)
        d["pred_prob_map_mtime"] = mtime(prob_map)

        native = Path(cfg["native_pred_mask"][stem])
        d["native_pred_mask_path"] = str(native)
        d["native_pred_mask_md5"] = md5(native)
        d["native_pred_mask_mtime"] = mtime(native)

        pred_quant = REPO / cfg["pred_quant_dir"][stem]
        subenc = pred_quant / "subendocardium.png"
        d["pred_subendocardium_png_mtime"] = mtime(subenc)

        pred_int = REPO / cfg["pred_intensity_dir"][stem]
        d["pred_band_means_csv"] = (pred_int / "band_means.csv").read_text() if (pred_int / "band_means.csv").exists() else None

        # chronological check: checkpoint save time < prob_map mtime < native mask mtime < quant output mtime
        ckpt_epoch = None
        for line in evidence["checkpoint_remote_stat"]:
            parts = line.split(" ", 1)
            if len(parts) == 2:
                ckpt_epoch = float(parts[0])
        # Prefer authoritative remote (laptop) timestamps -- see remote_mtime()
        # docstring for why local mtimes can be untrustworthy after a retried
        # transfer. Fall back to local mtime only if the remote file is gone
        # (e.g. already cleaned up) or laptop is unreachable.
        quant_rel = f"{cfg['pred_quant_dir'][stem]}/subendocardium.png"
        r_prob = remote_mtime(Path(cfg["pred_prob_map"][stem]).name)
        r_native = remote_mtime(Path(cfg["native_pred_mask"][stem]).name)
        r_quant = remote_mtime(quant_rel)

        prob_mtime = r_prob if r_prob is not None else (prob_map.stat().st_mtime if prob_map.exists() else None)
        native_mtime = r_native if r_native is not None else (native.stat().st_mtime if native.exists() else None)
        quant_mtime = r_quant if r_quant is not None else (subenc.stat().st_mtime if subenc.exists() else None)
        d["chronology_source"] = {
            "prob_map": "remote" if r_prob is not None else "local",
            "native_mask": "remote" if r_native is not None else "local",
            "quantification": "remote" if r_quant is not None else "local",
        }
        d["chronology_ok"] = bool(
            ckpt_epoch and prob_mtime and native_mtime and quant_mtime
            and ckpt_epoch < prob_mtime < native_mtime < quant_mtime
        )
        d["chronology_values_unix"] = {
            "checkpoint_saved": ckpt_epoch,
            "prob_map_generated": prob_mtime,
            "native_mask_generated": native_mtime,
            "quantification_generated": quant_mtime,
        }

        evidence["stems_detail"][stem] = d

    # 3. tool script checksums (proves same code path for gt and pred quantification)
    evidence["tool_script_md5"] = {
        "quantify_bands.py": md5(REPO / "quantify_bands.py"),
        "quantify_intensity.py": md5(REPO / "quantify_intensity.py"),
        "upsample_pred_mask.py": md5(REPO / "upsample_pred_mask.py"),
    }

    return evidence


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PAIRS:
        sys.exit(f"usage: python gather_evidence.py <{'|'.join(PAIRS)}>")
    pair = sys.argv[1]
    ev = gather(pair)
    outdir = REPO / "verification" / "evidence"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{pair}.json"
    outpath.write_text(json.dumps(ev, indent=2))
    print(f"wrote {outpath}")
    print(f"exclusion_verified: {ev['exclusion_verified']}")
    for stem, d in ev["stems_detail"].items():
        print(f"  {stem}: chronology_ok={d['chronology_ok']}")
