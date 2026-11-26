# Synchronized inference profiling

Run `tools.profile` after training and normal-resource construction. The default protocol uses training seed 3407, batch size one, the first 30 test images of each category in sorted file-path order, and the first 10 of those images for warm-up. Selection does not depend on labels. The command opens test images but never reads test masks or computes accuracy metrics.

## Inspect the input plan

```bash
python -m tools.profile --mode plan --dataset all --mvtec-root data/mvtec_anomaly_detection --visa-root data/VisA_pytorch/1cls --checkpoints-root outputs/training --resources-root outputs/resources --output outputs/profile_plan
```

The plan lists all 27 category jobs and hashes the selected input files. No model is loaded. Paths under `--checkpoints-root` and `--resources-root` follow the training and resource conventions in [reproduction.md](reproduction.md). For one category, explicit `--checkpoint` and `--resources` arguments override those roots.

## Measure one category or all 27

```bash
python -m tools.profile --mode run --dataset mvtec --category bottle --data-root data/mvtec_anomaly_detection --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --resources outputs/resources/mvtec/bottle/seed3407 --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --output outputs/profile_bottle --device cuda
python -m tools.profile --mode run --dataset all --mvtec-root data/mvtec_anomaly_detection --visa-root data/VisA_pytorch/1cls --checkpoints-root outputs/training --resources-root outputs/resources --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --output outputs/profile_all27 --device cuda
```

Use a fresh output directory for each run. For raw VisA, add `--layout raw --split-file data/VisA/split_csv/1cls.csv` when profiling VisA separately. Raw file-path ordering can differ from the prepared 1-class layout used for the paper's efficiency experiment; use the prepared layout to match that selection protocol.

The command sets OMP, MKL, OpenBLAS and PyTorch CPU thread counts to one. Before timing, it verifies resource hashes when a resource seal is available, loads the model and calibration, and compares instrumented and uninstrumented predictions on two normal training images. These two verification forwards do not enter the latency distribution. CUDA context creation and Python imports are also outside initialization timing.

## Timing stages

| Output key | Operations |
|---|---|
| `preprocess` | Input SHA-256 verification, image decoding/resizing and feature-input preprocessing |
| `host_to_device` | CPU-to-CUDA tensor transfers |
| `cnn_feature_response` | CNN feature extraction and raw response calculation, excluding nested preprocessing, transfers and prototype matching |
| `dino_feature` | Frozen DINOv2 feature extraction, excluding nested preprocessing and transfers |
| `matching_raw` | CNN normal-prototype matching and DINO global, same-position, neighborhood and CLS retrieval |
| `tail` | Response calibration, pixel fusion, top-1% aggregation and image scoring |
| `control_and_sync` | Remaining call control and synchronization overhead |

Each GPU stage has a `torch.cuda.synchronize` barrier before and after it. Every measured call also has start/end barriers. The timer uses `time.perf_counter_ns`; nested intervals contribute only their exclusive time. The seven nonoverlapping integer-nanosecond stages sum exactly to total call latency. Input verification, image I/O and the public API's CPU score/map copies are included.

`efficiency_classes.csv` reports per-category mean, median and 95th-percentile latency; reciprocal mean latency; stage means; PyTorch allocated/reserved peak memory; and initialization time. Each category directory contains `measured.json` and `timings.npz`. Memory peaks include resident tensors and are not total device memory measured by the operating system. CPU runs use `null` for CUDA memory. Reciprocal latency is not a separate measurement of pipelined throughput.

`efficiency_summary.json` takes equal-category means for latency and stage means, pools measured calls for median and 95th percentile, and reports maximum memory over categories. The paper's 27 categories have 30 measurements each, giving 810 measured calls, in addition to 270 warm-up and 54 normal verification forwards.

## Short installation check

```bash
python -m tools.profile --mode smoke --dataset mvtec --category bottle --input-split train --data-root data/mvtec_anomaly_detection --checkpoint outputs/training/mvtec/bottle/seed3407/train/best.pth --resources outputs/resources/mvtec/bottle/seed3407 --dino-repo third_party/dinov2 --dino-weights pretrained/dinov2_vits14_pretrain.pth --output outputs/profile_smoke --device cuda
```

Smoke mode measures two images after one overlapping warm-up image. `--input-split train` explicitly uses normal training images. It checks execution and timing accounting; it is not the 810-call paper benchmark. Both split and mode are recorded in the output.

## Figure 11 source data

`paper/plot_data/figure11_efficiency_classes.csv` and `figure11_efficiency_summary.json` preserve the recorded paper results. They are distinct from results newly generated by `tools.profile`. The historical implementation measured its synchronized evidence/debug API, including configuration reads and debug copies; the portable entry measures `PublicInference.predict`, which caches calibration and omits historical evidence-management work. The stage definitions, synchronization method and default sampling protocol are retained. Latency values also depend on hardware, library versions and available memory.

The development-stage synthetic response probes mentioned in the discussion are separate from Figure 11 and from the scoring-component controls. They are not additional accuracy or deployment benchmarks.
