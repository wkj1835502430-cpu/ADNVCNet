# Reproducing ADNVCNet

[Repository overview](../README.md)

## Installation

Use Python 3.12 and install matching PyTorch 2.6.0 and torchvision 0.21.0 builds for your hardware. The experiments used CUDA 11.8 builds. From the repository root:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-deps opencv-python==4.13.0.92
```

The separate OpenCV command preserves the NumPy 1.26.4/OpenCV 4.13 combination used by the numerical implementation. OpenCV is pinned because its rasterization and interpolation can affect pseudo anomalies and response maps. The other Python dependencies are recorded in `requirements.txt`.

Obtain [DINOv2](https://github.com/facebookresearch/dinov2) source in `third_party/dinov2` and its official **ViT-S/14 without registers** weights in `pretrained/dinov2_vits14_pretrain.pth`. The commands below pass both paths explicitly. CNN training uses the official torchvision ResNet-18 `IMAGENET1K_V1` initialization; torchvision downloads it on first use, or it can be supplied with `--imagenet-weights`.

Check the supplied DINO source and feature API before resource construction:

```bash
python -m tools.check_dino --repo third_party/dinov2 --feature-check --weights pretrained/dinov2_vits14_pretrain.pth --device cuda
```

`configs/dinov2_source_fingerprint.json` records the source-file hashes used for validation. Add `--strict` to require those exact source files. The feature check verifies the last-four-layer representation, the 37×37 patch grid, and the 384-dimensional CLS feature.

## Datasets

Obtain [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their providers. A prepared layout is:

```text
data/
  mvtec_anomaly_detection/
    bottle/train/good/...
    bottle/test/good/...
    bottle/test/<defect>/...
    bottle/ground_truth/<defect>/...
  VisA_pytorch/1cls/
    candle/train/good/...
    candle/test/good/...
    candle/test/bad/...
    candle/ground_truth/bad/...
```

Raw VisA is also supported. Pass `--layout raw --split-file /path/to/VisA/split_csv/1cls.csv` to training, resource construction, and prediction. For metrics and qualitative plots, pass the same `--split-file`. The CSV is available in the official VisA repository.

Normal image ordering and the reference/calibration splits are specified in `configs/resource_profiles.json`. The loader checks image hashes to apply the same protocol to raw and prepared layouts. Use the original dataset files without re-encoding. Training and calibration use normal training images; ground-truth masks are read only in evaluation or when explicitly requested for a qualitative plot.

## Run one category

The following example runs MVTec AD `bottle` with training seed 3407. Run all commands from the repository root. Each stage saves its inputs and resolved settings alongside the generated artifacts. Use a fresh output directory for a new run.

### 1. Train the CNN

```bash
python -m tools.train --dataset mvtec --category bottle --seed 3407 --data-root data/mvtec_anomaly_detection --output outputs/training --mode train --device cuda
```

The complete recipe uses 60 epochs and batch size 6. `best.pth` is selected by mean training loss. A plan can be inspected with `--mode plan`; `--mode smoke` checks one optimizer step on six normal images. Smoke outputs have a partial bank and are for checking the installation.

The full checkpoint path for this example is `outputs/training/mvtec/bottle/seed3407/train/best.pth`. Category-specific training settings and inference settings are stored separately in `configs/training_profiles.json`. See [training details](training.md) for the losses, initialization, and sampling recipe.

### 2. Construct normal references

```bash
python -m tools.prepare_resources --dataset mvtec --category bottle --seed 3407 --data-root data/mvtec_anomaly_detection --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --output outputs/resources/mvtec/bottle/seed3407 --device cuda --skip-calibration
```

This constructs the inference prototype bank, global and spatial DINO memories, the CLS memory, and the normal reference/calibration split. The resource directory is specific to the dataset, category, and CNN checkpoint.

### 3. Fit response calibration

```bash
python -m tools.calibrate --resources outputs/resources/mvtec/bottle/seed3407 --data-root data/mvtec_anomaly_detection --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --device cuda
```

Calibration uses normal A responses to fit the component, top-1%, and CLS mappings. CNN calibration excludes prototypes originating from the query image. Normal A component responses are also saved for the scoring controls below. Omitting `--skip-calibration` from step 2 performs steps 2 and 3 together.

### 4. Predict and evaluate

```bash
python -m tools.evaluate predict --dataset mvtec --category bottle --data-root data/mvtec_anomaly_detection --resources outputs/resources/mvtec/bottle/seed3407 --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --output outputs/predictions/mvtec/bottle/seed3407 --device cuda --save-components
python -m tools.evaluate metrics --predictions outputs/predictions/mvtec/bottle/seed3407 --data-root data/mvtec_anomaly_detection
```

Prediction finishes and seals the per-image scores and maps before metrics read the test annotations. `metrics.json` reports I-AUROC, P-AUROC, and AUPRO in percent. Pixel evaluation uses the saved continuous maps; AUPRO uses 200 thresholds, eight-connected regions, and an FPR limit of 0.30. `--save-components` also saves raw responses for scoring controls.

## Scoring controls

Use the same checkpoint and normal resources to isolate changes to the scoring stage:

```bash
python -m tools.ablate --resources outputs/resources/mvtec/bottle/seed3407 --predictions outputs/predictions/mvtec/bottle/seed3407 --data-root data/mvtec_anomaly_detection --output outputs/controls/mvtec/bottle/seed3407 --variants all
python -m tools.evaluate metrics --predictions outputs/controls/mvtec/bottle/seed3407/O1 --data-root data/mvtec_anomaly_detection
```

| Control | Change relative to A0 |
|---|---|
| A0 | Complete scoring method |
| A1 | Remove the CNN pixel response |
| A2 | Remove the DINO pixel responses |
| A3 | Remove the global DINO response |
| A4 | Remove the same-position and neighborhood responses, with the A4 fusion weights |
| A5 | Remove the CLS image score |
| A6 | Remove the top-1% image score |
| O1 | Aggregate each branch before weighted fusion |
| S_NO_S0 / S_NO_S1 / S_NO_BOTH | Remove spatial responses while retaining the remaining A0 coefficients |
| RAW_PIXEL | Use identity pixel mappings |
| LINEAR_Q_PIXEL | Use linear pixel mappings defined by normal 90th and 99.5th percentiles |
| TAIL_PIXEL | Use the empirical upper-tail pixel mappings |
| T1 | Replace all six response mappings by identity mappings |
| DINO_G | Score with the raw global DINO distance map and its top-1% mean |
| ID_CNN / ID_GLOBAL / ID_S0 / ID_S1 | Replace only the named pixel mapping by identity, then refit Top |
| ID_TOP / ID_CLS | Replace only the named image-score mapping by identity |

Controls that change calibrated pixel scores or their aggregation refit the corresponding Top mapping on normal A. A5 and A6 retain the corresponding A0 term; T1 and DINO_G use raw scores. Each control writes a new sealed prediction directory. Run `tools.evaluate metrics` on each directory to generate its metrics. A5, A6, O1, ID_TOP, and ID_CLS share the A0 pixel map and therefore have identical pixel localization metrics.

## All categories and three training seeds

For training, use `--category all` with `--dataset mvtec` or `--dataset visa`. Repeat with `--seed 3407`, `3408`, and `3409`. For VisA, use the VisA data root and category names. Run resource construction, calibration, prediction, metrics, and the selected controls for each trained category/seed using separate output directories.

Summarize the generated category metrics:

```bash
python -m tools.summarize --root outputs/predictions --output outputs/summary --require-all-categories
```

The script groups metrics by dataset, training seed, and variant. It first takes an equal-weight mean across unrounded category results, then reports the mean and sample standard deviation across seeds. `--require-all-categories` checks for 15 MVTec AD or 12 VisA categories in each group. Use `--root outputs/controls` to summarize the scoring controls separately.

## Generate figures

```bash
python -m tools.visualize qualitative --predictions outputs/predictions/mvtec/bottle/seed3407 --data-root data/mvtec_anomaly_detection --output outputs/experiment_figures/bottle_seed3407 --indices 0 1
python -m tools.visualize metrics --metrics outputs/predictions/mvtec/bottle/seed3407/metrics.json --output outputs/experiment_figures/bottle_metrics_seed3407
```

Qualitative plots show the input, anomaly map, and overlay. Add `--include-masks` to display ground truth after prediction verification. The default response display range is 0–10; it does not alter the saved scores. Plotting exports 600-dpi PNG and editable SVG files, together with their source paths or numeric data.

## Repository structure

```text
configs/     Category recipes and normal-data split definitions
datasets/    MVTec AD and VisA readers
models/      CNN training, DINO features, calibration, scoring, and metrics
tools/       Training, resource, prediction, evaluation, and plotting entry points
tests/       CPU checks for the numerical and workflow components
docs/        Training recipe
```

Run the CPU tests with `python -m unittest discover -s tests -v`. External models and datasets retain their original terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).

Repository: https://github.com/wkj-research/ADNVCNet
