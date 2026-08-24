# MAYO SEGMENTATION PAPER — FIGURE LIST

Methods paper, first author. Figure plan as supplied, annotated with what the
code and completed experiments can answer today.

**Contribution is the ROI prior mechanism.** Every figure explains it, proves it
works, or defends it.

---

## ANSWERS TO "THINGS TO RESOLVE BEFORE PLOTTING ANYTHING"

These were open in the plan. Four of five are now answered from the code, not
from memory. **One is a problem and needs a decision before any number is
quoted.**

### 1. Is IoU computed inside the ROI only, or over the whole slide? — **ROI-RESTRICTED. THIS NEEDS DISCLOSURE.**

`roi_exclusion_radius: 0.0` keeps only tiles whose bounding box overlaps the ROI:
**3,997 of 19,220 tiles (20.8%)**. The stitched canvas therefore covers a mean of
**23.8% of each slide** (range 5.2%–39.6%).

So 0.924 is IoU **over the ROI-adjacent region, not the whole slide.** Your
instinct was right and a reviewer will find it. Options:

- Report it as-is with the convention stated explicitly in Methods (defensible —
  it is the region the tool is asked to segment), **and**
- Additionally report whole-slide IoU, which requires re-running inference with
  `roi_exclusion: false`. Cheap, inference-only, and forecloses the objection.

Recommend doing both. Not yet run.

### 2. Validation or test? — **TEST.**

EXPERIMENT 11 is a genuine held-out test: per fold, 10 train / 3 validation /
1 test case. Early stopping and checkpoint selection read only the 3 validation
cases; the test case is scored once, after training. Use "test" everywhere.

The earlier 0.9515 figure (EXPERIMENT 9L) **was** validation and was
selection-contaminated — do not quote it.

### 3. Specimen-level split? — **YES, VERIFIED IN CODE.**

Stronger than specimen-level: **case-level**. 6 of the 20 slides are two physical
halves of one tissue sample and are always held out together (`CASE_GROUPS` in
`data_loader/data_loaders.py`), giving 14 cases. Verified at the loader: zero
stem overlap and zero case overlap between train/val, train/test and val/test.

Pairing was derived from **filenames, not directories** — pairs do not reliably
share a folder. Note `AJK056-10-2of2` + `AJKV056-10-1-of-2` are one case despite
the AJK/AJKV spelling difference.

### 4. Architecture specifics — **ANSWERED.**

| | |
|---|---|
| Backbone | ResNet-18, ImageNet-pretrained (`timm`, `pretrained=True`) |
| Encoder | **fully frozen** — 0 trainable params; all 3,231,617 trainable are the decoder |
| Input | 1 channel (grayscale); timm adapts the 3-ch stem |
| ROI encoding | binary mask, 1 channel, concatenated at the **bottleneck** (16×16) |
| Decoder | U-Net, widths 256/128/64/64/64, GroupNorm |
| Loss | `bce_dice`, a=0.3 · Adam 1e-4 · 800 epochs, early stop 100 |

⚠ Pin the exact timm weight tag before publishing (`resnet18.a1_in1k` vs the
torchvision port differ in recipe): `timm.create_model('resnet18',pretrained=True).default_cfg['tag']`

### 5. N — **ANSWERED.**

20 slides → **14 cases** · 19,220 tiles → **3,997** after ROI exclusion ·
**3 annotated ROI variants per slide** · single annotator (state as limitation).

---

## FIG 6 — ROI ROBUSTNESS ✅ **DONE**

`fig6_roi_robustness.png` · data `kfold_results/roi_robustness.json`

364 inference-only evaluations, 14 folds × 5 fixed annotator presets ×
6 strengths. No new data, no retraining.

| ROI displacement | IoU vs GT | vs baseline |
|---|---|---|
| 0 px | 0.9237 | — |
| 10 px | 0.9193 | −0.0044 |
| 22 px | 0.9072 | −0.0165 |
| 39 px | 0.8642 | −0.0595 |
| 70 px | 0.7637 | −0.1600 |
| 125 px | 0.6122 | −0.3115 |

**Graceful degradation, not a cliff** — which is the answer the objection needs.

**Critical caveat for the caption:** the real-annotator effect is NOT readable off
this curve. The three genuine variants differ by a median 40 px where the curve
reads −0.060, but *swapping real variants measures −0.0034* — a ~17× gap at
matched displacement, because a grid warp moves the ROI **off** the anatomy while
a second annotator still traces it. Present the sweep as a **conservative upper
bound**.

**Still to do for Fig 6, all cheap and inference-only:**
- [ ] Real-variant point on the same axes (`--real-variants`, implemented; 42 evals) — blocked only on gpu-box
- [ ] **Degenerate cases** — already implemented, just need running:
      - full-image ROI = `roi_jitter=1.0` (all-ones) ← *the strongest result if it still beats no-ROI*
      - empty ROI = `roi_dropout=1.0`
      - wrong-structure / random ROI — needs a few lines
- [ ] Translation / scale / rotation — **not implemented**; only grid warp exists

---

## FIG 4 — MAIN ABLATION ⚠ **NUMBERS DO NOT EXIST AT LOCO RIGOUR**

The ~0.50 / ~0.75 / ~0.90 progression comes from EXPERIMENT 4G/5A-era single-split
runs on a **smaller dataset**, and is **not comparable** to the 0.924 test number.
Only the full ROI model has been run through 14-fold LOCO.

To make this the money figure, the local-only and global-only arms must be re-run
under the same LOCO protocol. That is **training**, ~1.5 h/fold × 14 × 2 arms ≈ **2 days**.

Box plots across folds and a paired test (Wilcoxon signed-rank over the 14 paired
folds) are then straightforward — the per-fold data already exists for the ROI arm.

---

## STATUS OF THE REST

| Fig | Status | Blocker |
|---|---|---|
| 1 Pipeline | not started | needs wet-lab scope decision |
| 2 Dataset | **most stats answerable now** (see N above); needs split diagram + raw-image panel | none |
| 3 ROI mechanism | architecture answered above; diagram not drawn | none |
| 4 Ablation | ⚠ needs ~2 days retraining at LOCO rigour | GPU |
| 5 Qualitative | not started; stitching code exists (`slide_level_iou.py`) | none |
| 6 Robustness | ✅ **done**, extensions listed above | gpu-box for extras |
| 7 Failure cases | **candidates identified**: 17P645-1 (0.685, sectioning defect), FL_22P003-6 (0.898), AJKY121-7-2 (0.891) | none |
| 8 Data efficiency | not started; subsample + retrain | GPU |

### Supplementary — what already exists

- **S2 per-fold table** — done, in `kfold_results/` and the PDF report
- **S3 compute/runtime** — largely measured: 22 s/epoch on RTX 5090, ~1.5 h/fold, ~19 h for the full 14-fold sweep; inference ~15 min for all 14 folds
- **S6 annotation protocol** — single annotator; **3 ROI variants per slide exist**, so a genuine agreement number is computable (median 40 px boundary displacement between variants)
- **S1 hyperparameters** — all in `config.json`, transcribed above

### Negative results worth including

Three architectural levers were tested at full LOCO rigour and all failed. They
make the ceiling analysis credible rather than looking like unexplored options:

- **EXPERIMENT 12** decoder capacity — 18% of parameters reached within 0.0096
- **EXPERIMENT 13** neighbour context at the bottleneck — null at the noise floor
- **EXPERIMENT 14** capacity *placement* — −0.0050, val→test gap widened

And **EXPERIMENT 11A** decomposes the residual error: **41.2% boundary
imprecision**, 12.7% holes, 11.9% false-positive islands. Perfect topology
cleanup is worth at most **+0.013 IoU** — a hard ceiling that explains why the
topology-loss attempts (9F/9G/9H) failed.
