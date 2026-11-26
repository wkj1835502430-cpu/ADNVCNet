# Source-based training

The public training path reads the supplied normal training images, initializes
both ResNet-18 encoders with the same fixed ImageNet weights, constructs the
training prototype bank, and optimizes the student.

Run commands from the repository root. The default mode prints a plan:

```bash
python -m tools.train --dataset mvtec --category bottle --seed 3407 --mode plan
```

Check one actual optimizer step before a long run:

```bash
python -m tools.train --dataset mvtec --category bottle --seed 3407 --data-root /path/to/mvtec_anomaly_detection --output outputs/training_smoke --mode smoke --device cuda
```

This uses the first six normal images for a labeled one-epoch, one-batch smoke
run. Its prototype bank is partial. Its weights must not be used as a reported
full-training result. A complete run uses every normal training image:

```bash
python -m tools.train --dataset mvtec --category bottle --seed 3407 --data-root /path/to/mvtec_anomaly_detection --output outputs/training --mode train --device cuda
python -m tools.train --dataset visa --category candle --seed 3407 --data-root /path/to/VisA_pytorch/1cls --output outputs/training --mode train --device cuda
```

Use `--category all` for all categories of one dataset. Repeat with seeds 3408
and 3409 for the other two training runs. Each run writes its own
`<output>/<dataset>/<category>/seed<seed>/train/` directory and refuses to reuse
an existing directory. `best.pth` is selected by the minimum mean training loss;
`last.pth`, the history, resolved configuration, and normal-image list are saved
beside it. The checkpoint retains the complete original `model_state_dict` and
its configuration. It is accepted by the public resource-building entry point.

Raw VisA is also supported with the official 1-class CSV:

```bash
python -m tools.train --dataset visa --category candle --seed 3407 --data-root /path/to/raw/VisA --layout raw --split-file /path/to/VisA/split_csv/1cls.csv --output outputs/training --mode train --device cuda
```

Normal images are aligned to the canonical content-SHA order in
`configs/resource_profiles.json`, so raw and prepared layouts use the same
normal ordering and resource split. No test image or mask is opened by the
training reader. Use the original downloaded image files: re-encoding changes
their byte identities and will cause the canonical check to reject them.

## Exact category settings

`configs/training_profiles.json` contains all 27 category recipes. `model` describes training;
`evaluation_model` separately describes inference. Do not replace the training
settings with the fixed evaluation settings.

Training uses 60 epochs, batch size 6, float32, AdamW with learning rate and weight
decay 0.0001, no AMP, and a gradient-norm upper bound of 1e12. Training samples,
pseudo anomalies, and prototype sampling vary with the training seed; the
ImageNet initialization does not. The effective pseudo score is
`relu(1.75 * (pseudo_discrepancy - 0.8 * clean_discrepancy.detach()))`.
The probability is `clamp(1-exp(-8*score),1e-6,1-1e-6)`. Dice uses that
probability with smoothing 1. The synthetic mask is not dilated; copy-paste
mixing uses hard boundaries while preserving the original random draws, and
the appearance perturbation strength is 2.

All 27 training configurations use a discrepancy coefficient of 1. MVTec AD uses
category-specific prototype coefficients .43, .45, .48, or .50, together with
category-specific training layer weights, sampling limits, perturbation areas,
and loss coefficients. VisA uses a training prototype coefficient of .45. At
inference, both datasets use .35 and their fixed evaluation layer weights. The
training raw response contributes to response regularization; the supervised
pseudo score uses the discrepancy response alone.

Torchvision obtains the official ResNet-18 IMAGENET1K_V1 initialization from
`https://download.pytorch.org/models/resnet18-f37072fd.pth`. For offline use,
provide `--imagenet-weights /path/to/resnet18-f37072fd.pth`. The full SHA256 must
be `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
These are ImageNet initialization weights, not trained ADNVCNet weights.

## Environment and tests

The recorded environment uses Python 3.12.4, PyTorch 2.6.0+cu118, torchvision
0.21.0, NumPy 1.26.4, Pillow 10.3.0, and `opencv-python==4.13.0.92` (reported by
`cv2` as 4.13.0). Install matching PyTorch and torchvision builds for your
hardware, install `requirements.txt`, and then install OpenCV separately to
retain the recorded NumPy version:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps opencv-python==4.13.0.92
```

Keep the OpenCV version fixed: its ellipse rasterization contributes to
synthetic training masks. DINOv2 source and initialization are needed for
resource building and inference, not for CNN training.

Run the portable CPU tests with:

```bash
python -m unittest discover -s tests -p "test_public_training.py" -v
```
