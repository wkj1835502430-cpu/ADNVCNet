# Collect numerical figure inputs from a new run

`tools.collect_plot_data` converts newly computed category metrics and saved
analysis outputs into the CSV/JSON schemas read by `tools.paper_figures`.
It computes category means and differences from existing unrounded metrics;
it does not run inference, fit response mappings, recompute AUROC/AUPRO,
read masks, select a seed, or reuse the published numerical values.

## Complete workflow

Finish predictions and the separate `tools.evaluate metrics` stage before
collecting figure data. For Fig. 5, derive and evaluate A1–A6 and T1 in
addition to A0; T1 replaces all six response mappings with identity mappings.
For Fig. 6, also derive and evaluate O1. For Fig. 7, use
`tools.spatial_controls` to fit normal-only spatial mappings, generate A0,
S1_DOUBLE, and S0_S8 predictions, and evaluate their metrics separately.
Generate Fig. 8 statistics with `tools.uncertainty bootstrap` and Fig. 11
measurements with `tools.profile --mode run --dataset all`.

```bash
python -m tools.collect_plot_data \
  --metrics-root outputs/seed3407/main_predictions \
  --metrics-root outputs/seed3407/controls \
  --spatial-root outputs/seed3407/spatial_controls \
  --uncertainty-dir outputs/uncertainty/new_seed3407 \
  --profile-dir outputs/profile/all27 \
  --training-seed 3407 \
  --output-data outputs/new_plot_data

python -m tools.paper_figures \
  --data-dir outputs/new_plot_data \
  --figures 5 6 7 8 11 \
  --output outputs/new_figures
```

Both the collected data and figure output directories must be new. Paths are
examples; choose the directories where the actual commands wrote their
outputs. Use disjoint metric roots where possible. Repeating `--metrics-root`
or `--spatial-root` searches several roots, and overlapping references to the
same file are read once. Separate A0 metric files are deduplicated only when
their dataset, category, seed, image count, and all three unrounded metrics
are identical. A conflicting A0, or duplicate non-A0 category/variant, raises
an error rather than selecting a result.

The collector checks complete category sets: 15 MVTec AD categories and 12
VisA categories. It checks equal image counts for paired methods, exact
A0/A5/A6 shared-map pixel metrics, and exact A0/O1 pixel metrics. Spatial and
main A0 baselines must agree if both are supplied. There is no implicit
subset mode and no renormalization over whatever categories happen to exist.

## Select only the figures you have computed

By default, `--metrics-root` requests Figs. 5 and 6, `--spatial-root` requests
Fig. 7, `--uncertainty-dir` requests Fig. 8, and `--profile-dir` requests
Fig. 11. You may request a narrower explicit list with `--figures`.

For example, A0/O1 metrics alone can supply Fig. 6 without requiring unrelated
component controls:

```bash
python -m tools.collect_plot_data \
  --metrics-root outputs/seed3407/main_predictions \
  --metrics-root outputs/seed3407/order_controls \
  --figures 6 \
  --output-data outputs/order_plot_data

python -m tools.paper_figures --data-dir outputs/order_plot_data \
  --figures 6 --output outputs/order_figure
```

Each selected figure still requires both complete datasets. All requested
inputs are validated before any output files are written. Additional seeds
under a search root are ignored only when their declared `training_seed`
differs from the explicitly selected seed; the collector never averages
seeds or chooses the best seed.

## Output schemas

- `figure05_ablation.csv`: 42 rows, one for each dataset × seven controls
  (A1–A6/T1) × three metrics. Columns are `dataset`, `variant`, `metric`,
  `A0_minus_control_pp`, and `inherited_identical_pixel_map`. Each difference
  subtracts the control's equal-category macro metric from A0. Only the
  unchanged pixel metrics of A5/A6 receive the shared-map flag.
- `figure06_fusion_order.csv`: 27 category rows plus two `Macro mean` rows.
  Columns are `dataset`, `category`, and `joint_minus_separate_pp`. A category
  value is A0 I-AUROC minus O1 I-AUROC. The macro point is the equal-weight
  NumPy mean of the per-category differences.
- `figure07_spatial_macro.csv`: six rows for the two datasets × A0,
  S1_DOUBLE, and S0_S8. Columns are `dataset`, `variant`, `categories`,
  `image_auroc`, `pixel_auroc`, and `aupro`. Values are unrounded,
  equal-category means. The renderer subtracts the corresponding A0 macro.
- `figure08_intervals.csv`: byte-for-byte copy of
  `uncertainty_macro.csv`. The collector verifies the four contrasts, full
  category counts, interval nesting, and agreement with `results.json`.
  It requires a complete public uncertainty run using 10,000 resamples,
  resampling seed 20261028, and the four-comparison Bonferroni family.
- `figure11_efficiency_classes.csv` and
  `figure11_efficiency_summary.json`: byte-for-byte copies of the profiler
  files. The collector verifies all 27 unique categories, batch size 1,
  run mode, selected training seed, seven stage keys, finite GPU-memory
  measurements, stage sums, equal-category dataset means, and class/dataset
  peak-memory consistency. Smoke or partial profiling cannot supply a
  complete Fig. 11.
- `collection_manifest.json`: source and output SHA256 fingerprints,
  selected figures and seed, and explicit records of identical A0
  deduplication. Source locations are represented relative to numbered
  supplied roots.

The collector preserves unrounded values. All metric differences are in
percentage points; figure labels and rounding are applied by the renderer.
The original plot-data directory remains the separate source for regenerating
the published figures. A new output directory represents the results of the
newly selected run and may differ numerically and in measured latency.

## Software checks

```bash
python -m unittest tests.test_collect_plot_data -v
```

The checks use synthetic complete category fixtures to verify exact schemas,
counts, macro means, shared-map flags, identical A0 deduplication, unchanged
file copies, explicit figure selection, and rejection of missing categories,
conflicting runs, altered confidence intervals, and inconsistent timings.
