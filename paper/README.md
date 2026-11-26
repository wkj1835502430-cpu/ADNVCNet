# Manuscript figures and numerical sources

The `figures/` directory contains the exact English PNG figures used in manuscript v72. The files are linked to measured responses or numerical summaries below. `plot_data/` contains the original plot inputs with source hashes; the plotting command does not invent or refit experimental values.

| Figure | Content | Numerical input / computation |
|---|---|---|
| [1](figures/figure_01.png) | Cable responses and top-1% locations | Selected CNN/global response arrays and dataset-image identities in `qualitative_selection.json` |
| [2](figures/figure_02.png) | Overall method | Method schematic; editable SVG is supplied |
| [3](figures/figure_03.png) | Retrieval ranges and response calibration | Retrieval schematic plus all saved cable calibration knots in `figure03_calibration.npz`; command regenerates the calibration panel |
| [4](figures/figure_04.png) | Fusion and aggregation order | Method schematic using the same cable response maps as Figure 1; editable SVG is supplied |
| [5](figures/figure_05.png) | Component and mapping ablations | `figure05_ablation.csv`; A0 minus each control, in percentage points |
| [6](figures/figure_06.png) | Aggregation-order differences | `figure06_fusion_order.csv`; all 27 categories and both macro means |
| [7](figures/figure_07.png) | Fixed-total-weight spatial controls | `figure07_spatial_macro.csv`; each arm minus A0; [experiment commands](../docs/spatial_controls.md) |
| [8](figures/figure_08.png) | Paired stratified bootstrap intervals | `figure08_intervals.csv`; recompute from 3,887 image-score rows in `bootstrap_scores.csv` using [uncertainty commands](../docs/uncertainty.md) |
| [9](figures/figure_09.png) | MVTec AD qualitative results | Original continuous score arrays, fixed image/mask identities, and the official dataset |
| [10](figures/figure_10.png) | VisA qualitative results | Original continuous score arrays, fixed image/mask identities, and the official dataset |
| [11](figures/figure_11.png) | Latency and resource measurements | `figure11_efficiency_classes.csv` and `figure11_efficiency_summary.json`; [measurement commands](../docs/profiling.md) |

## Render the recorded numerical results

From the repository root:

```bash
python -m tools.paper_figures --output outputs/experiment_figures/recorded_results
```

This exports the Figure 3 calibration panel and Figures 5, 6, 7, 8, and 11 as 600-dpi PNG and editable SVG. The numeric content matches the recorded measurements; the exporter uses a portable Matplotlib layout. The manuscript PNGs preserve the exact author-edited layout.

To include the image-based Figures 1, 9, and 10, supply the official datasets:

```bash
python -m tools.paper_figures --figures all --mvtec-root data/mvtec_anomaly_detection --visa-root data/VisA_pytorch/1cls --output outputs/experiment_figures/all_empirical_panels
```

The selected original image and mask bytes are checked against their recorded hashes. Stored continuous response arrays remain unchanged; the 0–10 range clips the display only. Ground truth is displayed after prediction and is not an input to anomaly scoring.

Each export writes `plot_provenance.json`, including the numerical-source hashes, plotted values, and output hashes. Use `--data-dir` to render another run with the same documented CSV/NPZ schemas. Response arrays in the archived plot inputs are float64 and preserve their original numerical bytes.

## Recompute the confidence intervals

```bash
python -m tools.uncertainty bootstrap --scores paper/plot_data/bootstrap_scores.csv --output outputs/uncertainty/paper_scores
```

This reruns the 10,000 paired stratified resamples from the original frozen-model image scores. It reports the ordinary 95% intervals and the 98.75% intervals for the four Bonferroni-adjusted comparisons. To analyze newly generated predictions, use the prediction-root option in the [uncertainty guide](../docs/uncertainty.md).

## Dataset attribution

The example images visible in the manuscript figures are from [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff). Dataset images and masks used to rerender the qualitative panels are obtained from those providers and retain their dataset terms. This directory supplies computed response arrays, selected image identities, figure outputs, and numerical plot inputs; it does not redistribute the complete datasets or model weights.
