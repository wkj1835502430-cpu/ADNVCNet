# ADNVCNet

Source and experiment protocols for **Normal Variation Awareness and Response Calibration for Industrial Image Anomaly Detection**.

ADNVCNet combines a CNN teacher–student discrepancy and normal-prototype distance with frozen DINOv2 features. It retrieves normal references globally, at the same coordinate, and in a local neighborhood. Normal-response distributions calibrate the component scores before coordinate-wise fusion. Image scoring combines the mean of the highest 1% responses in the fused map with a CLS-feature score.

## Release scope and reproducibility status

This payload contains the core source, frozen configurations, small result summaries and numeric plotting data used for the manuscript. It excludes original dataset images, checkpoints, normal banks, DINO memories, calibration/prediction arrays, private machine audit inventories, IDE state and manuscript files. `source_manifest.json` records the SHA256 and size of every byte-exact source/result copy.

The source still refers to SHA-bound historical local resources and absolute Windows paths. It is **not a verified clean-machine, end-to-end release**. A source-only clone cannot currently reproduce the accepted results without either the accepted resource archive or a reviewed resource-construction implementation. Existing supervisors must be adapted in a separate, verified portability change; do not disable their integrity checks. Some supervisors may still resolve paths to the original installation.

No author-produced checkpoint/resource download has been published by this preparation. The accepted main inference resources total approximately 18.52 GiB and are excluded from this public scope. `docs/RESOURCE_RECONSTRUCTION.md` identifies recoverable constructors and the remaining integration/validation work.

## Contents

```text
models/                   CNN, DINO primitives, runtime and scoring
datasets/                 Data loaders, transforms and prototype sampling
losses/                   Compatibility modules retained from the project
tools/                    Main train/evaluate/resource-check entrypoints
  ablations/              Scoring, spatial, mapping and aggregation controls
  seeds/                  Two additional training seeds and three-seed reports
  baselines/              PatchCore/SimpleNet adapters and formal supervisor
  analysis/               Bootstrap, sensitivity, efficiency and figure exporters
configs/                  Actual main YAMLs and frozen experiment plans
archive/                  SHA-bound source/config snapshots and relocation maps
outputs/final_ablation_v1/ Minimal frozen A0–A6 executor source/config dependency
results_snapshot/         Small saved tables; no per-image predictions
docs/                     Entry map, resource list, environment and audit
third_party_licenses/     License texts inspected for external dependencies
source_manifest.json     Original path, destination, SHA256 and byte size
```

Archive entries are retained because the accepted runtime dynamically loads exact source/configuration bytes and the relocation layer resolves historical bindings. They are not a second set of recommended experiments. Office/Visio manipulation scripts, manuscripts, IDE state, historical development candidates, and cached third-party code are excluded.

## Environment

The recorded experiment environment used Windows, Python **3.12.4**, PyTorch **2.6.0+cu118**, torchvision **0.21.0+cu118**, CUDA runtime **11.8**, and an NVIDIA RTX 3090. `D:/anaconda3/python.exe` was the task interpreter; the project's PyCharm `.venv` was a different environment.

`environment/requirements-observed-main.txt` and `environment/requirements-observed-baselines.txt` record installed versions from distribution metadata. `docs/ENVIRONMENT.md` explains the additional user-site OpenCV installation and a dependency-metadata conflict. These are observations, not a fresh-install validation or a tested environment lock. DINOv2 and baseline source versions must also match their frozen source bindings.

## Data

Obtain **MVTec AD** from its rights holder and **VisA** from its official repository, subject to their licenses and access terms. Dataset images are not redistributed here. The experiments use 15 MVTec AD categories and the official VisA **1-class** layout with 12 categories. Training/resource fitting uses normal images; test labels and masks are reserved for evaluation after prediction sealing.

Historical roots were `G:/datasets/mvtec_anomaly_detection` and `G:/datasets/VisA_pytorch/1cls`. Merely changing these paths in a YAML is insufficient: the frozen contracts also bind image paths/hashes and validate configuration identity. See `docs/REPRODUCTION_STATUS.md` before adapting the package.

## Entrypoints

| Purpose | Entry |
|---|---|
| Main training | `tools/train_nvdr_fullclass.py` |
| Main evaluation / normal smoke / saved report | `tools/eval_adnvc_final.py` |
| Resource audit | `tools/check_adnvc.py` |
| A0–A6 saved-result report | `tools/ablations/eval_adnvc_final_ablation.py` |
| Original A0–A6 calculation | `outputs/final_ablation_v1/evidence/frozen_sources/eval_adnvc_final_ablation.py` plus `adnvc_ablation.py` and reference |
| Joint tail identity control | `tools/ablations/eval_adnvc_tail_ablation.py` |
| A0/O1 aggregation order | `tools/ablations/eval_adnvc_fusion_order_control.py` |
| Additive spatial removals / single-map identities / same-DINO baseline | `tools/ablations/eval_adnvc_scoring_isolation_v2.py` |
| Streaming metric continuation | `tools/ablations/eval_adnvc_scoring_metrics_stream_v3.py` |
| Three pixel mappings | `tools/ablations/eval_adnvc_pixel_mapping_control_v1.py` |
| Spatial center controls | `tools/ablations/eval_adnvc_spatial_center_control.py` |
| Seeds 3408 and 3409: training | `tools/seeds/train_adnvc_seed_repeats.py` |
| Seeds 3408 and 3409: normal gates and A0/A4/O1 evaluation | `tools/seeds/eval_adnvc_seed_stability_v1.py` |
| Three-seed report, including existing 3407 | `tools/seeds/report_adnvc_seed_stability_v1.py` |
| Actual formal PatchCore/SimpleNet comparison | `tools/baselines/eval_paper_external_formal_v1.py` |
| Bootstrap / leave-one-category-out summaries | `tools/analysis/analyze_adnvc_uncertainty.py`, `summarize_fusion_order_sensitivity.py` |
| Efficiency measurements | `tools/analysis/profile_adnvc_efficiency.py` |
| Saved empirical figures | `tools/analysis/export_experiment_figures.py` |
| Additional scoring-control figures | `tools/analysis/export_scoring_isolation_figures_v3.py` |

Exact flags, defaults, configuration constants, and original hardcoded path locations are in `the entry scripts`. The main training supervisor defaults to real training and the evaluation supervisor defaults to a real test run. The A0–A6 wrapper is report-only. Running a figure exporter does not recompute its underlying experiments.

The main method uses seed 3407; seeds 3408 and 3409 repeat CNN training while retaining the specified ImageNet initialization and fixed DINO resources. A0 is the complete scoring rule. A4 removes both spatial components and redistributes the retained weights. O1 aggregates each component before weighted combination; A0 fuses first and then aggregates. A0 and O1 share the pixel map. Additional fixed-coefficient spatial controls are separate from A4.

`train_adnvc_loss_ablation.py` is retained because seed training imports its helpers. Its historical 189-run loss-ablation plan was stopped and must not be reported as a completed experiment.

## Manuscript traceability

`docs/PAPER_ARTIFACT_MAP.md` maps all 18 tables and 11 figures in the current Chinese manuscript to the relevant scripts and saved-result locations. Figures 2 and 4 are method schematics. The curve in Figure 3 is computed from saved normal calibration data; schematic elements are drawn. Subsequent author edits to the Visio figures are not reproducible by the experimental plot exporter alone.

The accepted artifacts explicitly identify themselves as **post-development / reconstructed_not_original**. A new training run follows the accepted recipe but does not claim to recreate the original historical checkpoint bit for bit. The package preserves the existing experiment scope and does not turn post-development controls into independent blind tests.

## License and citation

No author-project license was present in the inspected source project. A license has not been assigned during packaging. Third-party dependencies retain their own terms; see `THIRD_PARTY_NOTICES.md`. Publication details and a repository citation can be added after they are confirmed.

Recovered normal-resource construction algorithms and the remaining integration steps are documented in `docs/RESOURCE_RECONSTRUCTION.md`. Historical builder references are preserved separately and are not active final-method entrypoints.
