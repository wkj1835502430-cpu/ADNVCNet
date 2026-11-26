# Image-score uncertainty and repeated-seed analysis

`tools/uncertainty.py` reproduces the statistical calculations in Section 4.4.
It uses saved scores or evaluated category metrics. It does not train models,
change response mappings, choose parameters, or open test masks.

## Recalculate the published confidence-interval figure

Run from the repository root:

```bash
python -m tools.uncertainty bootstrap \
  --scores paper/plot_data/bootstrap_scores.csv \
  --output outputs/uncertainty/published_scores
```

The supplied CSV contains the 3,887 image-level score triples used for the
published fixed-model analysis. These are retained experiment outputs, not
fresh predictions from a new training run. Each row contains the dataset,
category, fixed category index, image index, relative image identity, image
SHA256, binary evaluation label, A0/A4/O1 scores, and CNN training seed 3407.
Scores use round-trip decimal representations of their original binary64
values. Images, defect masks, pixel-score arrays, and model weights are not
included. Sorting or filtering rows is unnecessary: `image_index` explicitly
preserves the original within-category order.

This calculation has been checked against all original resampling counts,
integer AUROC numerators, and AUROC replicates for all 27 categories and all
10,000 resamples. Both the four macro interval rows and 54 exploratory
category interval rows match the retained analysis exactly.

## Analyse newly generated predictions

First generate A0 predictions using `tools.evaluate predict --save-components`,
derive A4 and O1 using `tools.ablate`, and run `tools.evaluate metrics` on each
method. Each category/method directory must contain `predictions.json`,
`prediction_seal.json`, its prediction NPZ files, and the default `metrics.json`.
Provide disjoint search roots containing one A0, A4, and O1 run per category.
Duplicate category/seed/variant runs are rejected rather than selected.

```bash
python -m tools.uncertainty bootstrap \
  --predictions-root outputs/seed3407/main_predictions \
  --predictions-root outputs/seed3407/controls \
  --mvtec-root /data/mvtec_anomaly_detection \
  --visa-root /data/VisA \
  --training-seed 3407 \
  --output outputs/uncertainty/new_seed3407
```

Every selected prediction manifest, input image hash, and prediction file hash
is verified before any labels are consumed. Image labels are then taken from
the separate post-seal `metrics.json` records, whose prediction-seal hashes
must match. The three methods must use the same CNN checkpoint and the same
ordered image identities and labels. The tool loads only their scalar image
scores from the NPZ files and confirms their AUROCs against the saved metrics.
It does not recompute pixel metrics or read mask contents. Input image bytes
are read solely to verify the prediction seals.

The default analysis requires all 27 categories. For an explicitly limited
software check, add `--allow-partial` and, optionally, `--replicates 137`.
Such outputs are marked as partial and are not a reproduction of the full
paper experiment. Changing the replicate count changes the original PCG64
draw sequence because complete stratum matrices are drawn in order; a smaller
run is not a prefix of the 10,000-resample run.

## Exact statistical protocol

1. Hold the trained models, normal references, calibration, fusion weights,
   and saved image scores fixed. Analyse A0, A4, and O1 jointly.
2. Enumerate VisA categories alphabetically at indices 0–11, followed by
   MVTec AD categories alphabetically at indices 12–26. These indices remain
   fixed even in a partial analysis. The order is declared as `CASE_ORDER`.
3. For each category, create `Generator(PCG64(SeedSequence([20261028,
   case_index])))`. Draw all normal-image resamples first, then all
   anomalous-image resamples, with replacement and unchanged stratum sizes.
   The original implementation uses int16 resampling counts and rejects
   strata of 32,767 images or more.
4. Use the same per-image counts for A0, A4, and O1. Calculate AUROC with exact
   integer normal/anomalous pair wins; an equal-score pair contributes half
   a win. Convert each category AUROC to percent after integer accumulation.
5. In each replicate, average category AUROCs with equal category weights
   within each dataset, then subtract A4 or O1 from A0. Categories themselves
   are not resampled. Images from different seeds are not pooled.
6. Use NumPy linear quantiles at 0.025/0.975 for ordinary 95% percentile
   intervals. For the four macro comparisons—two contrasts on each of two
   datasets—use 0.00625/0.99375 for 98.75% intervals, corresponding to the
   Bonferroni adjustment and approximately 95% joint coverage. The family
   size remains four in an explicitly partial software run.

The intervals quantify image-sampling uncertainty for the existing categories
and fixed trained models under an image-independence assumption. They do not
measure training variability or unseen-category performance. Category-level
95% intervals are exploratory and are not adjusted for multiple comparisons.
The original point estimates were already known when this post-hoc analysis
plan was fixed; the bootstrap is not an independent validation experiment.

## Statistical output and the confidence-interval figure

Each invocation requires a new output directory. It writes:

- `analysis_plan.json`: seed, ordered cases, interval definition, and input
  fingerprints, saved before calculating new intervals.
- `results.json`, `uncertainty_macro.csv`, `uncertainty_per_category.csv`:
  unrounded statistics. Macro rows use `dataset`, `comparison` (`A0-A4` or
  `A0-O1`), `delta_I_AUROC_pp`, `ci95_low_pp`, `ci95_high_pp`,
  `familywise95_bonferroni_low_pp`, and `familywise95_bonferroni_high_pp`.
  These fields directly supply the points and intervals for Fig. 8.
- One directory per category containing `paired_inputs.npz`,
  `resample_counts.npz`, `replicates.npz`, `input_identity.json`, and
  `record.json`. The replicate arrays preserve integer numerators and the
  corresponding three-column AUROC values for numerical checks.

All differences and interval bounds are in percentage points. A0 is
fuse-then-aggregate; O1 is aggregate-then-fuse with its own aggregate-score
mapping; A4 jointly removes both spatial references, redistributes fusion
weights, and refits its aggregate-score mapping. The contrasts retain these
definitions rather than attributing the entire difference to one operation.

## Three-seed paired differences and leave-one-category summaries

```bash
python -m tools.uncertainty paired \
  --metrics-root outputs/seed3407 \
  --metrics-root outputs/seed3408 \
  --metrics-root outputs/seed3409 \
  --output outputs/uncertainty/three_seed_paired
```

The command reads existing unrounded public `metrics.json` files for
A0/A4/O1. It requires the same categories for all three variants and seeds
3407, 3408, and 3409. It first computes equal-category macro metrics per seed,
then reports their mean and sample standard deviation across seeds (`ddof=1`,
denominator 2 for three seeds). Paired differences are calculated within each
seed before their mean and sample SD are reported. A0 and O1 must have
identical pixel metrics because their localization map is shared.

For the A0–O1 image-level contrast, the tool additionally removes each category
once from the saved-category macro summary, reports every remaining-category
mean difference, and records the full range and sign counts. It uses
`statistics.fmean` on the per-category differences, as in the original
descriptive sensitivity analysis. This is reaggregation of fixed results:
there is no retraining, leave-one-category-out cross-validation, new AUROC
calculation, confidence-interval calculation, or GT I/O in this command.

Output files are `results.json`, `per_seed_macro.csv`,
`macro_mean_sample_sd.csv`, `paired_macro_differences.csv`,
`all_category_differences.csv`, `all_leave_one_summaries.csv`, and
`macro_sensitivity_summary.csv`. All positive, negative, and tied results and
all category exclusions are retained. `--allow-partial` supports an explicitly
limited fixture; at least two categories and complete pairing are still
required. The SD is null if that fixture has only one seed.

## Software checks

```bash
python -m unittest tests.test_uncertainty -v
```

The checks cover tie-heavy AUROC against independently expanded resamples,
the original PCG64 draw sequence, linear interval endpoints, binary64 CSV
round trips, fixed category indices, complete pairing, verification before
label consumption, sample SD, and all leave-one-category exclusions.
