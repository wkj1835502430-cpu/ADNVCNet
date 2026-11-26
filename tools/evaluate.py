"""Predict first, seal predictions, then evaluate masks in a separate command.

Subcommands:
  predict    Run frozen inference without opening any test mask.
  metrics    Verify the prediction seal, then read labels and masks.
  aggregate  Compute equal-category macro means from saved category metrics.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from datasets.public_data import IMAGE_SUFFIXES, canonical_dataset, _visa_csv
from tools.public_utils import read_json, write_json_new, sha256, now, relative_file, load_profile


def discover_images(data_root, dataset, category, layout='auto', split_file=None):
    """Return ordered test-image paths; do not read labels or open masks."""
    root = Path(data_root).resolve()
    prepared = root / category / 'test'
    if layout == 'prepared' or (layout == 'auto' and prepared.is_dir()) or dataset == 'mvtec':
        if not prepared.is_dir():
            raise FileNotFoundError(prepared)
        images = sorted(p for p in prepared.glob('*/*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        return [p.relative_to(root).as_posix() for p in images], 'prepared'
    records = []
    with _visa_csv(root, split_file).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['object'].strip() == category and row['split'].strip().lower() == 'test':
                rel = row['image'].strip().replace('\\', '/')
                relative_file(root, rel)
                records.append(rel)
    return sorted(records), 'raw'


def predict(args):
    from models.public_inference import PublicInference, dataset_name
    dataset = canonical_dataset(args.dataset)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Prediction output already exists; select a new directory')
    profile = load_profile(args.resources)
    if dataset_name(profile) != dataset:
        raise ValueError('Resource profile does not match --dataset')
    declared_category = profile.get('dataset', {}).get('category') if isinstance(profile.get('dataset'), dict) else profile.get('category')
    if declared_category and declared_category != args.category:
        raise ValueError('Resource profile does not match --category')
    inputs, layout = discover_images(args.data_root, dataset, args.category, args.layout, args.split_file)
    if not inputs:
        raise ValueError('No test images found')
    runtime = PublicInference(args.checkpoint, profile, args.dino_repo, args.dino_weights, args.resources, args.device)
    if runtime.calibration is None:
        raise FileNotFoundError('calibration.npz is required for test inference')
    output.mkdir(parents=True)
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir()
    rows = []
    for index, relative in enumerate(inputs):
        image = relative_file(args.data_root, relative)
        input_digest = sha256(image)
        raw = runtime.raw_components(image)
        result = runtime.score_raw(raw)
        if args.save_components:
            values = dict(result)
            values.update({'raw_' + key: value for key, value in raw.items()})
        else:
            values = {name: result[name] for name in ('anomaly_map', 'image_score', 'pixel_top1')}
        target = prediction_dir / f'{index:06d}.npz'
        with target.open('xb') as stream:
            np.savez_compressed(stream, **values)
        rows.append({'input': relative, 'input_sha256': input_digest,
                     'prediction': target.relative_to(output).as_posix(), 'prediction_sha256': sha256(target)})
        print(f'PREDICTED {index + 1}/{len(inputs)} {relative}', flush=True)
    resources = {name: sha256(args.resources / name) for name in ('profile.json', 'bank.npz', 'memories.npz', 'calibration.npz')}
    manifest = {'schema': 'adnvcnet_public_predictions_v1', 'dataset': dataset, 'category': args.category,
                'training_seed': int(profile['training_seed']), 'variant': 'A0',
                'layout': layout, 'rows': rows, 'count': len(rows), 'GT_masks_read': False,
                'checkpoint_sha256': sha256(args.checkpoint), 'resource_sha256': resources,
                'save_components': args.save_components, 'device': args.device,
                'run_identity': 'new_user_generated_predictions', 'completed_at': now()}
    write_json_new(output / 'predictions.json', manifest)
    write_json_new(output / 'prediction_seal.json', {
        'schema': 'adnvcnet_public_prediction_seal_v1', 'manifest': 'predictions.json',
        'manifest_sha256': sha256(output / 'predictions.json'), 'count': len(rows),
        'GT_masks_read': False, 'sealed_at': now()})
    print(f'PREDICTIONS_SEALED {output}', flush=True)


def verify_predictions(folder, data_root):
    folder = Path(folder).resolve()
    seal = read_json(folder / 'prediction_seal.json')
    if seal.get('schema') != 'adnvcnet_public_prediction_seal_v1' or seal.get('GT_masks_read') is not False:
        raise ValueError('Missing or invalid pre-metric prediction seal')
    manifest_path = relative_file(folder, seal['manifest'])
    if sha256(manifest_path) != seal['manifest_sha256']:
        raise ValueError('Prediction manifest changed after sealing')
    manifest = read_json(manifest_path)
    if manifest.get('GT_masks_read') is not False or len(manifest['rows']) != seal['count']:
        raise ValueError('Prediction count or GT flag mismatch')
    for row in manifest['rows']:
        image = relative_file(data_root, row['input'])
        prediction = relative_file(folder, row['prediction'])
        if sha256(image) != row['input_sha256'] or sha256(prediction) != row['prediction_sha256']:
            raise ValueError('A sealed input or prediction file changed')
    return manifest


def ground_truth_rows(data_root, manifest, split_file=None):
    """This is called only after every prediction hash has been verified."""
    root = Path(data_root).resolve()
    if manifest['layout'] == 'raw':
        lookup = {}
        with _visa_csv(root, split_file).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['object'].strip() == manifest['category'] and row['split'].strip().lower() == 'test':
                    lookup[row['image'].strip().replace('\\', '/')] = row
        result = []
        for row in manifest['rows']:
            item = lookup[row['input']]
            label_text = item['label'].strip().lower()
            if label_text not in ('normal', 'anomaly', 'anomalous', 'abnormal'):
                raise ValueError(f'Unknown VisA label: {label_text}')
            label = int(label_text != 'normal')
            mask = relative_file(root, item['mask'].strip().replace('\\', '/')) if label else None
            result.append((label, mask))
        return result
    result = []
    for row in manifest['rows']:
        image = relative_file(root, row['input'])
        label = int(image.parent.name != 'good')
        mask = None
        if label:
            candidates = [root / manifest['category'] / 'ground_truth' / image.parent.name / (image.stem + suffix)
                          for suffix in ('_mask.png', '.png', image.suffix)]
            mask = next((p for p in candidates if p.is_file()), None)
            if mask is None:
                raise FileNotFoundError(f'No defect mask found for {row["input"]}')
        result.append((label, mask))
    return result


class ScoreSequence:
    """Reiterable lazy loading: at most one prediction map is materialized."""
    def __init__(self, folder, rows):
        self.folder, self.rows = Path(folder), rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        with np.load(self.folder / self.rows[index]['prediction'], allow_pickle=False) as data:
            value = np.asarray(data['anomaly_map'], dtype=np.float64).copy()
        if value.shape != (512, 512) or not np.isfinite(value).all():
            raise ValueError('Expected a finite 512x512 prediction map')
        return value
    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class MaskSequence:
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        label, mask = self.rows[index]
        if not label:
            return np.zeros((512, 512), dtype=np.float32)
        with Image.open(mask) as image:
            value = (np.asarray(image.convert('L')) > 0).astype(np.float32)
        return cv2.resize(value, (512, 512), interpolation=cv2.INTER_NEAREST)
    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def metrics(args):
    from models.public_metrics import disk_pixel_auc, threshold_aupro, roc_auc_score
    manifest = verify_predictions(args.predictions, args.data_root)
    target = args.output if args.output else args.predictions / 'metrics.json'
    if target.exists():
        raise FileExistsError('Metrics already exist; do not overwrite an evaluated run')
    truth = ground_truth_rows(args.data_root, manifest, args.split_file)
    score_maps = ScoreSequence(args.predictions, manifest['rows'])
    masks = MaskSequence(truth)
    image_scores = []
    for row in manifest['rows']:
        with np.load(args.predictions / row['prediction'], allow_pickle=False) as data:
            value = float(data['image_score'])
            if not math.isfinite(value):
                raise ValueError('Nonfinite image score')
            image_scores.append(value)
    labels = [r[0] for r in truth]
    if len(set(labels)) != 2:
        raise ValueError('AUROC requires both normal and anomalous test images')
    work = args.predictions / 'metrics_work'
    work.mkdir(exist_ok=False)
    scratch = work / 'sorted_negative_pixels.npy'
    pixel_auc, pixel_proof = disk_pixel_auc(score_maps, masks, scratch)
    aupro = threshold_aupro(score_maps, masks, max_fpr=.30, thresholds=200)
    image_auc = float(roc_auc_score(np.asarray(labels, dtype=np.uint8), np.asarray(image_scores, dtype=np.float64)))
    row = {'schema': 'adnvcnet_public_metrics_v1', 'dataset': manifest['dataset'], 'category': manifest['category'],
           'training_seed': int(manifest['training_seed']), 'variant': manifest.get('variant', 'A0'),
           'count': len(labels), 'image_auroc': image_auc * 100., 'pixel_auroc': pixel_auc * 100.,
           'aupro': aupro * 100. if math.isfinite(aupro) else None,
           'unit': 'percent', 'max_fpr': .30, 'thresholds': 200, 'connectivity': 8,
           'continuous_scores_used': True, 'pixel_sampling': False, 'pixel_auc_proof': pixel_proof,
           'prediction_seal_sha256': sha256(args.predictions / 'prediction_seal.json'),
           'ground_truth': [{'label': label, 'mask_sha256': sha256(mask) if mask else None} for label, mask in truth],
           'run_identity': 'new_user_generated_metrics', 'completed_at': now()}
    write_json_new(target, row)
    # Only remove the scratch array created by this successful invocation.
    if scratch.resolve().parent == work.resolve():
        scratch.unlink()
    print(f'METRICS_COMPLETE {manifest["dataset"]} {manifest["category"]} {image_auc*100:.6f} {pixel_auc*100:.6f} {aupro*100:.6f}', flush=True)


def aggregate(args):
    rows = [read_json(path) for path in args.metrics]
    seen = set()
    groups = {}
    for row in rows:
        key = row['dataset'], row['category']
        if key in seen:
            raise ValueError('Duplicate category metric; aggregate each training seed separately')
        seen.add(key)
        groups.setdefault(row['dataset'], []).append(row)
    result = {}
    for dataset, values in groups.items():
        if any(row.get(metric) is None for row in values for metric in ('image_auroc', 'pixel_auroc', 'aupro')):
            raise ValueError('Cannot macro-average an undefined category metric')
        result[dataset] = {'categories': [v['category'] for v in values],
                           'macro': {metric: float(np.mean([v[metric] for v in values])) for metric in ('image_auroc', 'pixel_auroc', 'aupro')}}
    write_json_new(args.output, {'schema': 'adnvcnet_public_macro_metrics_v1', 'datasets': result,
                   'aggregation': 'equal-category macro mean, using unrounded category metrics',
                   'unit': 'percent', 'completed_at': now()})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('predict', help='Create and seal predictions without opening GT masks')
    p.add_argument('--dataset', choices=('mvtec', 'visa'), required=True)
    p.add_argument('--category', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dino-repo', type=Path, required=True)
    p.add_argument('--dino-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layout', choices=('auto', 'prepared', 'raw'), default='auto')
    p.add_argument('--split-file', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--save-components', action='store_true', help='Also retain raw and calibrated components for score ablations')
    m = sub.add_parser('metrics', help='Verify sealed predictions, then compute test metrics')
    m.add_argument('--predictions', type=Path, required=True)
    m.add_argument('--data-root', type=Path, required=True)
    m.add_argument('--split-file', type=Path)
    m.add_argument('--output', type=Path)
    a = sub.add_parser('aggregate', help='Compute category macro means from saved metrics')
    a.add_argument('--metrics', nargs='+', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    {'predict': predict, 'metrics': metrics, 'aggregate': aggregate}[args.stage](args)


if __name__ == '__main__':
    main()
