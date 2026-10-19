# Baselines and revised quantification

Three comparison arms for the ventricle segmentation paper, and a revision to how
the downstream measurement and its agreement statistic are computed.

Everything below is measured, not assumed; the numbers are from runs on all 20
slides unless a smaller n is stated. Work currently lives in `/tmp` and has to be
promoted into this folder — see **To implement**.

---

## 1. The arms

| arm | what it is | mean IoU (n=20) | (n=19, no 17P645-1) |
|---|---|---|---|
| ROI upsampled | the published floor, `roi-upsample.json` | 0.843 | 0.839 |
| microSAM | ViT-H histopathology generalist | 0.785 | 0.782 |
| SAM | ViT-H, prompted with the model's ROI | 0.883 | 0.880 |
| **ROI − tissue** | `ROI AND NOT tissue`, no network | 0.867 | **0.894** |
| model | 14-fold LOCO, published | 0.936 | 0.939 |

All scored at native TIFF resolution through `slide_level_metrics.score_native`,
so they are directly comparable to the published model metric.

**The finding that matters.** Excluding `17P645-1`, a one-line morphological rule
*beats* SAM at its maximum supported resolution (0.894 vs 0.880), and beats it on
9 of 9 human sections. SAM's apparent advantage over it rests on a single
pathological slide. The model clears both by +0.045 with a third the spread.

**microSAM scores below the ROI floor on 18 of 20 sections**, so it is a negative
control, not a competing method. Report it in text and in the statistics CSV; do
not give it figure space.

### SAM / microSAM configuration

Whole slide, not tiles: SAM is an object segmenter and a 512 px tile is an
arbitrary crop of an object. Its encoder is fixed at `img_size=1024` — confirmed
from the checkpoints themselves, `pos_embed` is (1, 64, 64, 1280) — so 1024 on
the long side is its maximum supported resolution, and the mask is upsampled ~34x
to native against the model's ~4.3x.

Prompt is `roi_<stem>.pt` whole and unaltered: (1, 256, 256) on the padded square
canvas, which is exactly SAM's dense mask-prompt frame, so it transfers with no
resampling. Every component including single-pixel ones; nothing thresholded out.

Two free parameters, both set from a **plateau** rather than an optimum, which is
what keeps them from being fitted on the evaluation set:

- `--mask-logit 20`. Mask-only prompting is unusable — swept over scale it ran
  0.00 to 0.91 IoU on one slide and peaked at a different scale on each of three,
  because SAM's prompt encoder was trained on mask logits accompanied by a sparse
  prompt. With points added it is flat to within 0.015 over logit 12–30.
- `--point-spacing 16` ROI px, a lattice over the ROI interior plus one
  guaranteed point per component at its distance-transform maximum. One point per
  component is too sparse: microSAM returns a nucleus-sized mask, which is its
  prior. From spacing 32 inward IoU moves under 0.03 while the point count
  changes thirtyfold.

No box prompt: these ROIs are arcs and a component fills a median 28% of its own
bounding box.

Intensity: SAM's API takes uint8 and has no learned gain, while our encoder takes
float32 and lets BatchNorm set it. Exposure varies ~5x between sections
independently of source dtype. One scalar linear gain per slide puts the 99.9th
percentile over scanned pixels at 255 — **one** scalar across R, G and B (per
channel would launder the real MYL4/counterstain imbalance), applied in the
source dtype with the 8-bit cast last (gaining after a full-scale conversion
recovers nothing). Reaches 11.7x on `17P645-1`, 1.1x on `AJKY121-8-2`.

### Why the baselines fail, mechanistically

At nearly identical **area** the boundary **lengths** differ enormously:

| slide | arm | area/GT | perimeter/GT |
|---|---|---|---|
| AJKV056-7 | model | 0.916 | 0.938 |
| | SAM | 0.947 | **0.525** |
| | ROI−tissue | 0.858 | 0.642 |

The depth rings are grown outward from the mask boundary, so ring area is roughly
boundary length x ring width. A boundary half as long samples half the near-field
tissue — and the tissue it misses is the MYL4-rich subendocardial rim. The
baselines therefore regress every slide toward the bulk value: ground truth spans
0.495–0.606 in the pig near band while SAM spans 0.475–0.573, giving Pearson
r = −0.11 (SAM) and −0.36 (trivial). They do not track between-slide variation.

**This is not fixable by tuning the morphology.** Ablating `close_mask`, the
tissue island filter and the prediction island filter moves perimeter over a
sixfold range (0.64 to 3.99) while IoU moves under 0.02 on every slide. You can
dial the trivial rule's boundary to exactly the annotation's length
(`AJKV056-7`, no-tissue-islands: perimeter 1.022) and gain nothing — the extra
boundary is threshold noise, not anatomy. The model's advantage is boundary
*placement*, not boundary *quantity*.

**Also note:** the model's two worst near-band slides have IoU 0.954 and 0.970,
among its best segmentations. High IoU and poor near-band agreement coexist, so
IoU should not be the headline metric.

---

## 2. Measurement: per-pixel instead of ratio-of-sums

Published definition, a ratio of sums over band AND tissue:

    rho_sum = sum(G) / (sum(G) + s * sum(R))

New definition, every pixel one vote:

    rho_px  = mean over pixels of  g / (g + s * r)

`s` is **kept**. It is the G:R ratio over the tissue outside the mask and cancels
the per-slide channel gain; per-pixel ratios are *not* invariant to a per-channel
gain. Dropping it sends one section from 0.55 to 0.27 — that variant is computed
as `rho_px_raw` so the cost is visible rather than assumed.

**Why change.** `rho_sum` is intensity-weighted, so a few bright pixels can carry
a band, and the `>200 um` band is *pinned* near 0.5 by construction because it
holds almost all the pixels and the normaliser forces it there.

**What it costs.** The biological effect roughly halves — pig near-band minus
deep-band goes from 0.109 to 0.059 — and the measure is fragile at low counts
(see the exclusion below).

**What it does not change.** The ordering is model > SAM > ROI−tissue in every
band and species under both definitions. Publish both if space allows; the
supplement showing an unchanged ordering is the strongest possible answer to
"did you pick the metric that flattered you".

---

## 3. Agreement: Bland–Altman instead of CCC

**CCC was dropped.** It divides by the between-slide spread, so it is a
signal-to-noise ratio, not an agreement measure, and it is **not comparable
across two measurement definitions**. On this data, identical masks scored
CCC 0.837 under one definition and 1.000 under the other; and the `>200 um` band
read 0.07 for an arm whose absolute agreement was the best in the figure. A
reviewer is right to distrust a variance-normalised headline.

**Report instead**, per band per arm: bias, 95% limits of agreement, and mean
absolute difference as a share of the biological effect (near band minus deep
band, measured on the annotation for that species).

| | bias | LoA width | % of effect |
|---|---|---|---|
| Pig 0–99 model | −0.019 | 0.075 | **30%** |
| Pig 0–99 SAM | −0.038 | 0.207 | 76% |
| Pig 0–99 ROI−tissue | −0.051 | 0.284 | **107%** |
| Human 0–99 model | −0.001 | 0.046 | **10%** |
| Human 0–99 SAM | −0.005 | 0.090 | 24% |
| Human 0–99 ROI−tissue | −0.001 | 0.124 | 32% |
| Human 100–199 model | +0.003 | 0.016 | 6% |
| Human 100–199 SAM | +0.017 | 0.042 | 29% |

The trivial rule's limits of agreement are **wider than the effect being
measured**. This cannot be moved by rescaling, and it separates two failures CCC
conflated: in pig the baselines are roughly unbiased but wildly imprecise; in
human 100–199 they are precise but biased.

**Phrase the baseline result as "does not track between-slide variation
(r = −0.11)", not "CCC = −0.07".** The latter reads as worse-than-chance, which
is not the finding.

**The `>200 um` band carries no agreement statistic.** It is bulk tissue, barely
touches the mask, and every arm agrees there by construction — CCC 1.000 under
per-pixel, 0.07 under ratio-of-sums, both meaningless. Plot it for the depth
gradient, label it boundary-insensitive. Do **not** describe the near band as
noisy; it is the only band that discriminates methods.

---

## 4. Exclusions

- `AJKY121-8-2` — **excluded** under the per-pixel measure, not marked. Its MYL4
  channel is near zero, so `g/(g+sr)` is a mean over ratios of noise and lands at
  0.03, stretching the axis and contributing nothing measurable. Marking suited a
  ratio of sums; here the quantity does not exist on that section.
- `17P391_7A`, `17P645-1` — plotted with their own markers, not scored, matching
  the published figure's convention.
- This leaves pig n = 11 plotted / 9 scored, human n = 8 plotted / 8 scored.

Note the published `density-by-depth.pdf` marks rather than excludes
(`EXCLUDED = ()`). If the baseline figure excludes, say so in the caption; the
two figures otherwise use different conventions and a reader will notice.

---

## 5. To implement in the repo

Currently in `/tmp`, needs promoting into `figures-for-pub/sam-baseline/`:

| from | to | notes |
|---|---|---|
| `/tmp/roi_minus_tissue/quantify_trivial.py` | `trivial_baseline.py` | writes `masks/trivial/<stem>.npz`, `trivial.json` (IoU), in `score_native` format |
| `/tmp/perpixel/measure.py` | `quantify_arms.py` (replace) | one image pass, all four arms, emits `rho_sum`/`rho_px`/`rho_px_raw` |
| `/tmp/perpixel/figure_ba.py` | `figure_baselines.py` | Bland–Altman annotation, per-pixel measure |

Already in place and correct: `export_sam_masks.py`. Delete `figure_sam_baseline.py`
(CCC-based, superseded) and `quantify_arms.py`'s current CCC-era form.

Repoint every `/tmp` path to `HERE`. `masks/` is already gitignored
(`.gitignore:140`).

Checks that must stay, because they are what makes the arms comparable:

- `quantify_arms.py` re-derives the ground-truth arm per slide and compares it to
  `quantification/density.csv`. It currently reports `d = 0.00e+00` on all 20.
  If that ever becomes non-zero, the tissue mask or ring geometry has moved and
  the baseline rows are not comparable to the published ones.
- `export_sam_masks.py --arm stub-roi` runs the whole harness with the ROI
  standing in for SAM; its IoU must reproduce `roi-upsample.json`'s per-slide
  `roi` value (currently within 1.3e-2, the residual being 256→1024→native versus
  256→8192→native).
- The figure aborts if any arm is missing a slide rather than plotting a short
  series.

---

## 6. Blocked on the k-fold rerun

`17P645-1`'s mask and ROI are being redone. The blast radius:

**The ROI change reaches the model, not just SAM.** `roi_exclusion` selects which
tiles the model ever sees, so a new `roi_17P645-1.pt` changes the tile set, the
stitched canvas and the model's prediction. If only SAM gets the new ROI the arms
stop being comparable.

**The GT change reaches published numbers.** Every arm is scored against
`<stem>_mask.*`, so the model's IoU in
`k-fold-baseline-2026-09-02_slide_metrics.json`, the `roi`/`ceiling`/`model`
triple in `roi-upsample.json`, and the ground-truth rho in
`quantification/density.csv` all move — which regenerates `density-by-depth.pdf`,
the main result.

Order:

1. Confirm the new mask TIFF matches the image dimensions. `quantify_density`
   asserts it and `to_native` assumes it; a mismatch otherwise surfaces three
   steps later as something confusing.
2. `utils/slide_level_metrics.py --name k-fold-baseline --outdir
   figures-for-pub/model-baseline --save-masks quantification/predictions` — the
   expensive step; it loops all 14 folds with no `--slides` filter.
3. `figures-for-pub/roi-upsample/roi_upsample.py`
4. `quantification/quantify_density.py` — all 20, or merge a filtered run
5. `export_sam_masks.py --arm sam` and `--arm microsam`, `trivial_baseline.py`
6. `quantify_arms.py`, then both figures

Steps 2 and 4 dominate; everything else is minutes.

---

## 7. Open items

- **`CLIP_PCT = 95` is untested.** It caps the bright tail before Otsu and the
  docstring says a few saturated pixels otherwise drag the threshold from 5 to
  46. It is the one genuinely hand-set parameter, and it feeds the tissue mask
  that *every* arm's rings are intersected with — so it is already baked into the
  published rho. The morphology ablation covered close radius and both island
  fractions, not this. Worth a sweep before publication.
- **The ring-area explanation is only half-confirmed.** Perimeter sets ring area,
  but matching perimeter demonstrably does not improve the mask. Whether it
  improves *agreement* is untested; the decisive run is rho on the
  perimeter-matched trivial variant.
- **Figure layout:** the two stat blocks sit ~0.5 in apart and can read as one
  run-on block. A light vertical rule between bands, or one shared column header,
  fixes it.
- **ROI − tissue as a deliverable.** It reaches 0.894 IoU with no training, no
  GPU and no tunable parameters that matter (IoU moves under 0.02 across every
  morphology setting). That makes it a credible annotation *seed* for
  out-of-distribution material, which is worth a sentence. Its failure mode must
  be stated: when thresholding cannot separate cavity from tissue it degrades to
  a hollow rim (`17P645-1`, IoU 0.337), so it needs a glance before use.
