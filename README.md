# ADNVCNet

**Industrial Image Anomaly Detection with Normal Variation Awareness and Response Calibration**

ADNVCNet detects and localizes industrial image anomalies using normal training images. It combines convolutional teacher–student feature discrepancies, distances to normal prototypes, and normal-reference retrieval with frozen DINOv2 features.

## Method

1. **Local representations.** A convolutional teacher–student branch measures local feature deviations. A frozen DINOv2 branch extracts patch and whole-image features.
2. **Normal references.** Patch features are matched against global, same-position, and neighborhood reference sets to account for appearance and structural position.
3. **Response calibration.** Each response is calibrated against its distribution on normal samples before the response maps are aligned and fused.
4. **Anomaly scoring.** The fused map provides pixel-level localization. Its top-1% mean is combined with a CLS-feature score for image-level detection.

## Reproduction workflow

The computational workflow comprises dataset preparation, normal-reference construction, CNN training, normal-response calibration, and test evaluation. Experiment settings include the category-specific training parameters, reference sampling rules, fusion weights, and scoring definitions.

The experiments use the 15 categories of MVTec AD and the official 12-category VisA 1-class split. Normal images are used for training and reference construction. Test labels and masks are used for evaluation.

## Experiments

The paper studies normal-reference ranges, response calibration, and fusion/aggregation order. Repeated CNN training uses seeds 3407, 3408, and 3409 with the specified pretrained initialization and DINO representation. Evaluation reports image-level AUROC, pixel-level AUROC, and AUPRO.

Training, evaluation, analysis, and figure-generation commands produce their outputs in the corresponding output directories. Historical experiment outputs are kept separate from the source release.

## Data and attribution

MVTec AD and VisA are obtained from their original providers under their respective terms. Pretrained backbones and third-party components retain their original licenses and attribution requirements.
