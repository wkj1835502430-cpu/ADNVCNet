# Fixed-total-weight spatial controls (Figure 7)

These controls distinguish the center location from the surrounding positions
while keeping the total spatial coefficient at one third. Run them with the
category models trained using seed 3407, after constructing normal resources,
fitting calibration, and saving A0 predictions with `--save-components`.

| Arm | Pixel map after component calibration and interpolation |
|---|---|
| A0 | `C/2 + (G + S0 + S1)/6` |
| S1_DOUBLE | `C/2 + (G + S1 + S1)/6` |
| S0_S8 | `C/2 + (G + S0 + S8)/6` |

`S8` searches the eight offsets of the 3-by-3 neighborhood excluding the
center. Candidates outside the feature grid are omitted. The implementation
retains the dataset-specific dot-product precision: float32 for MVTec AD and
float64 for VisA, with float32 best-similarity accumulation. Retrieval uses the
same frozen DINO features and spatial memory as the main method.

## 1. Fit on normal A

From the repository root, using MVTec AD bottle as an example:

```bash
python -m tools.spatial_controls fit --resources outputs/resources/mvtec/bottle/seed3407 --data-root data/mvtec_anomaly_detection --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --output outputs/spatial_fits/mvtec/bottle/seed3407 --device cuda
```

The command reads the original ordered normal A component responses from
`calibration_raw/` and computes S8 for the same images. It first fits the S8
mapping, then the Top mapping of each arm. The other component mappings and
CLS mapping remain fixed. The refitted A0 Top parameters must exactly match
the original calibration. `normal.json`, `fits.npz`, saved S8 responses, and
`normal_seal.json` record these normal-only inputs and parameters.

## 2. Predict and seal all three arms

```bash
python -m tools.spatial_controls predict --resources outputs/resources/mvtec/bottle/seed3407 --data-root data/mvtec_anomaly_detection --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --fits-dir outputs/spatial_fits/mvtec/bottle/seed3407 --predictions outputs/predictions/mvtec/bottle/seed3407 --output outputs/spatial_predictions/mvtec/bottle/seed3407 --device cuda
```

This stage verifies the source A0 prediction seal, reuses its raw components,
and computes the additional DINO retrieval response. It checks
`S1 = min(S0, S8)` within `1e-6`, checks S0/S1 retrieval directly on the first
and last inputs, and verifies that the A0 pixel map is unchanged. It performs
no CNN forward pass and reads no test labels or masks. The three arm folders
use the standard `predictions.json` and `prediction_seal.json` format.

## 3. Evaluate and summarize

Run metrics separately on each arm after the prediction command finishes:

```bash
python -m tools.evaluate metrics --predictions outputs/spatial_predictions/mvtec/bottle/seed3407/A0 --data-root data/mvtec_anomaly_detection
python -m tools.evaluate metrics --predictions outputs/spatial_predictions/mvtec/bottle/seed3407/S1_DOUBLE --data-root data/mvtec_anomaly_detection
python -m tools.evaluate metrics --predictions outputs/spatial_predictions/mvtec/bottle/seed3407/S0_S8 --data-root data/mvtec_anomaly_detection
```

Repeat for all 15 MVTec AD and 12 VisA categories, using the corresponding data
root, category checkpoint, resources, and fresh output folders. For raw VisA,
pass the same official `--split-file` to the metrics command as in the main
workflow. Then compute the equal-category means:

```bash
python -m tools.summarize --root outputs/spatial_predictions --output outputs/spatial_summary --require-all-categories
```

Use the resulting per-dataset arm means to plot each arm minus A0, in
percentage points, in the order A0, S1_DOUBLE, S0_S8. The two non-A0 arms are
distinct from simply removing spatial terms without redistributing weights.
