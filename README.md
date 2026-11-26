# ADNVCNet

Code for **Industrial Image Anomaly Detection with Normal Variation Awareness and Response Calibration**.

ADNVCNet combines CNN teacher–student discrepancies with global, same-position, and neighborhood retrieval of frozen DINOv2 features. Normal-response calibration and pixel-wise fusion support defect localization and image-level detection.

## Installation

Python 3.12, PyTorch 2.6.0, and torchvision 0.21.0. For the CUDA 11.8 environment used in the experiments:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-deps opencv-python==4.13.0.92
```

## Data and pretrained models

- Datasets: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and the [VisA 1-class split](https://github.com/amazon-science/spot-diff).
- Backbone: official [DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2) source and weights without registers. ResNet-18 ImageNet initialization is obtained through torchvision.

## Usage

Follow the [reproduction guide](docs/reproduction.md) for data paths and complete commands, from training to evaluation.

| Task | Entry point |
|---|---|
| Train | [tools/train.py](tools/train.py) |
| Build normal references and calibrate | [tools/prepare_resources.py](tools/prepare_resources.py) |
| Predict and evaluate | [tools/evaluate.py](tools/evaluate.py) |
| Run scoring controls | [tools/ablate.py](tools/ablate.py) |
| Summarize results | [tools/summarize.py](tools/summarize.py) |
| Generate figures | [tools/visualize.py](tools/visualize.py) |

Generated checkpoints, predictions, metrics, and figures are saved to `outputs/`.

See [training details](docs/training.md) for category-specific settings and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.
