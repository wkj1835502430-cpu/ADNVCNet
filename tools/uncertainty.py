"""Reanalyse saved image scores: paired bootstrap, paired seeds, and category sensitivity.

This module never trains a model, selects a configuration, or opens a test mask.
The bootstrap keeps the original PCG64 draw order and integer tie-half AUROC.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path, PurePosixPath
from statistics import fmean
import numpy as np

from datasets.public_data import MVTEC_CATEGORIES, VISA_CATEGORIES
from tools.public_utils import now, read_json, relative_file, sha256, write_json_new

METHODS = ('A0', 'A4', 'O1')
METRICS = ('image_auroc', 'pixel_auroc', 'aupro')
SEEDS = (3407, 3408, 3409)
CASE_ORDER = tuple(('visa', c) for c in VISA_CATEGORIES) + tuple(('mvtec', c) for c in MVTEC_CATEGORIES)
CASE_INDEX = {key: i for i, key in enumerate(CASE_ORDER)}
SCORE_COLUMNS = ('dataset', 'category', 'case_index', 'image_index', 'input', 'input_sha256',
                 'label', 'A0', 'A4', 'O1', 'training_seed')


def validate(scores, labels, weights=None):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError('Expected finite one-dimensional scores and both binary labels')
    if weights is not None and (weights.ndim != 2 or weights.shape[1] != len(labels)
                                or weights.dtype.kind not in 'iu' or np.any(weights < 0)):
        raise ValueError('Invalid nonnegative integer bootstrap weights')


def doubled_wins(scores, labels, weights):
    """Exact integer Mann–Whitney numerator: two wins plus one for each tie."""
    validate(scores, labels, weights)
    pi, ni = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    ni = ni[np.argsort(scores[ni], kind='stable')]
    negatives = scores[ni]
    lo = np.searchsorted(negatives, scores[pi], side='left')
    hi = np.searchsorted(negatives, scores[pi], side='right')
    prefix = np.pad(np.cumsum(weights[:, ni], axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    return np.sum(weights[:, pi].astype(np.int64) * (prefix[:, lo] + prefix[:, hi]), axis=1, dtype=np.int64)


def direct_auc(scores, labels):
    """Image AUROC in percent, evaluated with exact win and tie counts."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    validate(scores, labels)
    positive, negative = scores[labels == 1, None], scores[labels == 0][None, :]
    return float((2 * np.count_nonzero(positive > negative) + np.count_nonzero(positive == negative))
                 / (2 * positive.size * negative.size) * 100)


def weights_for(labels, replicates, seed):
    """Retain the paper's entire-stratum draw order; do not batch these draws."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels.tolist()) != {0, 1} or int(replicates) != replicates or replicates < 1:
        raise ValueError('Invalid binary labels or replicate count')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    weights = np.zeros((replicates, len(labels)), np.int16)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if not 0 < len(indices) < np.iinfo(np.int16).max:
            raise ValueError('Stratum size exceeds the original int16-count protocol')
        draws = indices[rng.integers(0, len(indices), size=(replicates, len(indices)))]
        np.add.at(weights, (np.arange(replicates)[:, None], draws), 1)
    return weights


def quantile_interval(values, alpha):
    return np.quantile(values, [alpha / 2, 1 - alpha / 2], method='linear').tolist()


def _binding(path):
    path = Path(path)
    return {'file': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size}


def _csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty statistical table')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _check_relative(value):
    p = PurePosixPath(value)
    if not value or '\\' in value or ':' in value or p.is_absolute() or '..' in p.parts:
        raise ValueError('Image identities must be relative POSIX paths')
    return value


def validate_cases(cases, allow_partial=False):
    seen = set()
    seeds = set()
    for case in cases:
        key = (case['dataset'], case['category'])
        if key not in CASE_INDEX or key in seen or case['case_index'] != CASE_INDEX[key]:
            raise ValueError('Duplicate, unknown, or incorrectly indexed category')
        seen.add(key)
        seeds.add(case['training_seed'])
        scores, labels = np.asarray(case['scores']), np.asarray(case['labels'])
        if scores.shape != (len(labels), 3):
            raise ValueError('Scores must have columns A0, A4, O1')
        for j in range(3):
            validate(scores[:, j], labels)
        if len(case['inputs']) != len(labels) or len(set(case['inputs'])) != len(labels):
            raise ValueError('Missing or duplicated image identity')
        for value in case['inputs']:
            _check_relative(value)
    if not cases or len(seeds) != 1:
        raise ValueError('The fixed-model bootstrap requires one training seed')
    if not allow_partial and seen != set(CASE_ORDER):
        raise ValueError('Paper analysis requires all 27 categories; use --allow-partial only for a smoke/subset analysis')
    return sorted(cases, key=lambda c: c['case_index'])


def load_scores_csv(path):
    """Load round-trip binary64 scores, retaining the explicit original image order."""
    groups = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(SCORE_COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError('Scores CSV is missing required columns: ' + ', '.join(SCORE_COLUMNS))
        for row in reader:
            key = row['dataset'], row['category']
            groups.setdefault(key, []).append(row)
    cases = []
    for (dataset, category), rows in groups.items():
        rows.sort(key=lambda r: int(r['image_index']))
        if [int(r['image_index']) for r in rows] != list(range(len(rows))):
            raise ValueError('Each category needs consecutive unique image_index values starting at zero')
        if len({r['case_index'] for r in rows}) != 1 or len({r['training_seed'] for r in rows}) != 1:
            raise ValueError('Category metadata changed within a scores CSV')
        labels = np.array([int(r['label']) for r in rows], dtype=np.int64)
        cases.append(dict(dataset=dataset, category=category, case_index=int(rows[0]['case_index']),
            training_seed=int(rows[0]['training_seed']), inputs=[_check_relative(r['input']) for r in rows],
            input_sha256=[r['input_sha256'] for r in rows], labels=labels,
            scores=np.array([[float(r[name]) for name in METHODS] for r in rows], dtype=np.float64)))
    return cases


def _labels_from_metrics(folder, manifest):
    """Use labels already recorded by the separate post-seal metrics command."""
    path = Path(folder) / 'metrics.json'
    metric = read_json(path)
    if (metric.get('schema') != 'adnvcnet_public_metrics_v1'
            or metric.get('prediction_seal_sha256') != sha256(Path(folder) / 'prediction_seal.json')
            or any(metric.get(k) != manifest.get(k) for k in ('dataset', 'category', 'training_seed', 'variant'))
            or metric.get('count') != len(manifest['rows']) or metric.get('unit') != 'percent'):
        raise ValueError('Metrics do not belong to the sealed predictions: ' + str(path))
    labels = np.asarray([r['label'] for r in metric['ground_truth']])
    if labels.shape != (len(manifest['rows']),) or set(labels.tolist()) != {0, 1}:
        raise ValueError('Metrics require ordered labels for both test strata')
    return labels, metric, _binding(path)


def load_prediction_cases(roots, data_roots, training_seed=3407):
    """Verify every selected prediction before consuming any evaluation labels."""
    from tools.evaluate import verify_predictions
    selected = {}
    files = sorted({p.resolve() for root in roots for p in Path(root).rglob('predictions.json')})
    for path in files:
        metadata = read_json(path)
        if metadata.get('variant') not in METHODS or metadata.get('training_seed') != training_seed:
            continue
        dataset = metadata['dataset']
        if dataset not in data_roots or data_roots[dataset] is None:
            raise ValueError('Supply --' + dataset + '-root to verify sealed input hashes')
        manifest = verify_predictions(path.parent, data_roots[dataset])
        key = dataset, manifest['category'], manifest['variant']
        if key in selected:
            raise ValueError('Duplicate category/variant; select narrower prediction roots: ' + str(key))
        selected[key] = (path.parent, manifest)
    cases, bindings = [], []
    keys = sorted({(ds, category) for ds, category, _ in selected}, key=lambda k: CASE_INDEX[k])
    for dataset, category in keys:
        if any((dataset, category, name) not in selected for name in METHODS):
            raise ValueError('Each category needs sealed A0, A4, and O1 predictions')
        manifests = [selected[dataset, category, name] for name in METHODS]
        ordered = [[(r['input'], r['input_sha256']) for r in m['rows']] for _, m in manifests]
        if any(order != ordered[0] for order in ordered[1:]):
            raise ValueError('Paired methods must use the same ordered image identities')
        if len({m['checkpoint_sha256'] for _, m in manifests}) != 1:
            raise ValueError('A fixed-model comparison requires the same CNN checkpoint')
        columns, all_labels = [], []
        for folder, manifest in manifests:
            labels, metric, binding = _labels_from_metrics(folder, manifest)
            values = []
            for row in manifest['rows']:
                with np.load(relative_file(folder, row['prediction']), allow_pickle=False) as saved:
                    values.append(float(saved['image_score']))
            values = np.asarray(values, dtype=np.float64)
            if abs(direct_auc(values, labels) - float(metric['image_auroc'])) > 1e-12:
                raise ValueError('Saved image scores disagree with their evaluated AUROC')
            columns.append(values)
            all_labels.append(labels)
            bindings.append(dict(dataset=dataset, category=category, variant=manifest['variant'],
                manifest=_binding(folder / 'predictions.json'), seal=_binding(folder / 'prediction_seal.json'), metrics=binding))
        if not all(np.array_equal(all_labels[0], y) for y in all_labels[1:]):
            raise ValueError('Paired evaluation labels differ')
        cases.append(dict(dataset=dataset, category=category, case_index=CASE_INDEX[dataset, category],
            training_seed=training_seed, inputs=[r[0] for r in ordered[0]], input_sha256=[r[1] for r in ordered[0]],
            labels=all_labels[0], scores=np.stack(columns, axis=1)))
    return cases, bindings


def bootstrap(cases, output, *, replicates=10000, seed=20261028, allow_partial=False, sources=None):
    cases = validate_cases(cases, allow_partial)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / 'analysis_plan.json', dict(schema='adnvcnet_bootstrap_plan_v1',
        methods=METHODS, replicates=replicates, seed=seed, training_seed=cases[0]['training_seed'],
        case_order=[{'dataset': c['dataset'], 'category': c['category'], 'case_index': c['case_index']} for c in cases],
        family_macro_comparisons=4, ordinary_confidence=.95, adjusted_confidence=.9875,
        RNG='PCG64 SeedSequence([seed, fixed_case_index]); normal stratum, then anomalous stratum',
        quantile_method='linear', category_resampling=False, paired_methods=True,
        posthoc=True, fixed_before_this_computation=True, sources=sources or [], created_at=now()))
    per_category, evidence, grouped, observed = [], [], {}, {}
    for case in cases:
        dataset, category = case['dataset'], case['category']
        scores, labels = np.asarray(case['scores'], np.float64), np.asarray(case['labels'])
        folder = output / (dataset + '_' + category)
        folder.mkdir()
        weights = weights_for(labels, replicates, [seed, case['case_index']])
        numerator = np.stack([doubled_wins(scores[:, j], labels, weights) for j in range(3)], axis=1)
        denominator = 2 * int((labels == 0).sum()) * int((labels == 1).sum())
        auc = numerator / denominator * 100
        point = np.array([direct_auc(scores[:, j], labels) for j in range(3)])
        grouped.setdefault(dataset, []).append(auc)
        observed.setdefault(dataset, []).append(point)
        for j, name in ((1, 'A4'), (2, 'O1')):
            interval = quantile_interval(auc[:, 0] - auc[:, j], .05)
            per_category.append(dict(dataset=dataset, category=category, comparison='A0-' + name,
                count=len(labels), normal=int((labels == 0).sum()), abnormal=int((labels == 1).sum()),
                delta_I_AUROC_pp=float(point[0] - point[j]), ci95_low_pp=interval[0], ci95_high_pp=interval[1],
                exploratory_not_multiple_comparison_adjusted=True))
        _npz(folder / 'paired_inputs.npz', scores=scores, labels=labels)
        _npz(folder / 'resample_counts.npz', weights=weights)
        _npz(folder / 'replicates.npz', doubled_wins=numerator, I_AUROC=auc)
        write_json_new(folder / 'input_identity.json', dict(inputs=case['inputs'],
            input_sha256=case.get('input_sha256'), score_columns=METHODS, masks_opened=False))
        record = dict(dataset=dataset, category=category, count=len(labels), seed=[seed, case['case_index']],
            denominator=denominator, point_I_AUROC=point.tolist(),
            files={name: _binding(folder / name) for name in ('paired_inputs.npz', 'resample_counts.npz', 'replicates.npz', 'input_identity.json')})
        write_json_new(folder / 'record.json', record)
        evidence.append(dict(dataset=dataset, category=category, record=(folder / 'record.json').relative_to(output).as_posix(),
            sha256=sha256(folder / 'record.json')))
        print(f'BOOTSTRAP_CATEGORY {dataset} {category} n={len(labels)} replicates={replicates}', flush=True)
    macro = []
    for dataset in ('visa', 'mvtec'):
        if dataset not in grouped:
            continue
        auc, point = np.mean(np.stack(grouped[dataset]), axis=0), np.mean(np.stack(observed[dataset]), axis=0)
        for j, name in ((1, 'A4'), (2, 'O1')):
            delta = auc[:, 0] - auc[:, j]
            ci, adjusted = quantile_interval(delta, .05), quantile_interval(delta, .05 / 4)
            macro.append(dict(dataset=dataset, comparison='A0-' + name, categories=len(grouped[dataset]),
                delta_I_AUROC_pp=float(point[0] - point[j]), ci95_low_pp=ci[0], ci95_high_pp=ci[1],
                familywise95_bonferroni_low_pp=adjusted[0], familywise95_bonferroni_high_pp=adjusted[1],
                adjusted_interval_excludes_zero=bool(adjusted[0] > 0 or adjusted[1] < 0)))
    result = dict(schema='adnvcnet_public_uncertainty_v1', macro=macro, per_category=per_category,
        replicates=replicates, seed=seed, training_seed=cases[0]['training_seed'], methods=METHODS,
        family_macro_comparisons=4, unit='percentage_points', records=evidence,
        partial_analysis=bool(set((c['dataset'], c['category']) for c in cases) != set(CASE_ORDER)),
        sources=sources or [], new_model_forwards=0, new_training=0, mask_reads=0,
        pixel_metrics_recomputed=False, scope='Fixed-model image-sampling uncertainty conditional on existing categories and independent images; not training variability or external validation',
        completed_at=now())
    write_json_new(output / 'results.json', result)
    _csv(output / 'uncertainty_macro.csv', macro)
    _csv(output / 'uncertainty_per_category.csv', per_category)
    return result


def load_category_metrics(roots):
    rows, bindings = [], []
    files = sorted({p.resolve() for root in roots for p in Path(root).rglob('metrics.json')})
    for path in files:
        row = read_json(path)
        if row.get('schema') != 'adnvcnet_public_metrics_v1' or row.get('variant') not in METHODS:
            continue
        if row.get('unit') != 'percent' or any(row.get(k) is None or not math.isfinite(row[k]) for k in METRICS):
            raise ValueError('Expected finite unrounded category metrics in percent')
        rows.append(row)
        bindings.append(dict(dataset=row['dataset'], category=row['category'], training_seed=row['training_seed'],
                             variant=row['variant'], metrics=_binding(path)))
    return rows, bindings


def paired_summary(rows, output, *, allow_partial=False, sources=None):
    lookup = {}
    for row in rows:
        key = row['dataset'], row['category'], int(row['training_seed']), row['variant']
        if key in lookup or key[:2] not in CASE_INDEX or key[2] not in SEEDS or key[3] not in METHODS:
            raise ValueError('Duplicate or unsupported category, seed, or variant')
        if any(row.get(k) is None or not math.isfinite(row[k]) for k in METRICS):
            raise ValueError('Undefined or nonfinite category metric')
        lookup[key] = row
    if not lookup:
        raise ValueError('No paired category metrics found')
    complete_keys = {(ds, c, s, name) for ds, c in CASE_ORDER for s in SEEDS for name in METHODS}
    if not allow_partial and set(lookup) != complete_keys:
        raise ValueError('The paper summary requires all 27 categories, three seeds, and three variants')
    macros, across, paired, category_delta, leave_one, sensitivity = [], [], [], [], [], []
    for dataset in ('visa', 'mvtec'):
        categories = sorted({key[1] for key in lookup if key[0] == dataset})
        if not categories:
            continue
        seeds = sorted({key[2] for key in lookup if key[0] == dataset})
        expected = VISA_CATEGORIES if dataset == 'visa' else MVTEC_CATEGORIES
        if not allow_partial and (tuple(categories) != expected or tuple(seeds) != SEEDS):
            raise ValueError('The paper summary requires all categories and three seeds')
        if len(categories) < 2:
            raise ValueError('Leave-one-category analysis requires at least two categories')
        for training_seed in seeds:
            for name in METHODS:
                try:
                    selected = [lookup[dataset, c, training_seed, name] for c in categories]
                except KeyError as exc:
                    raise ValueError('All paired seeds and variants require identical categories') from exc
                macros.append(dict(dataset=dataset, seed=training_seed, variant=name, categories=len(categories),
                    **{metric: float(np.mean([r[metric] for r in selected])) for metric in METRICS}))
            deltas = {}
            for category in categories:
                base = lookup[dataset, category, training_seed, 'A0']
                order = lookup[dataset, category, training_seed, 'O1']
                if any(base[k] != order[k] for k in ('pixel_auroc', 'aupro')):
                    raise ValueError('A0 and O1 must retain identical pixel-map metrics')
                deltas[category] = float(base['image_auroc']) - float(order['image_auroc'])
                for control in ('A4', 'O1'):
                    for metric in METRICS:
                        category_delta.append(dict(dataset=dataset, seed=training_seed, category=category,
                            comparison='A0-minus-' + control, metric=metric,
                            delta_pp=float(base[metric]) - float(lookup[dataset, category, training_seed, control][metric])))
            values = []
            for excluded in categories:
                value = fmean(deltas[c] for c in categories if c != excluded)
                leave_one.append(dict(dataset=dataset, seed=training_seed, excluded_category=excluded,
                    retained_categories=len(categories) - 1, delta_pp=value))
                values.append(value)
            sensitivity.append(dict(dataset=dataset, seed=training_seed, categories=len(categories),
                full_delta_pp=fmean(deltas.values()), leave_one_min_pp=min(values), leave_one_max_pp=max(values),
                positive_categories=sum(v > 0 for v in deltas.values()), negative_categories=sum(v < 0 for v in deltas.values()),
                tied_categories=sum(v == 0 for v in deltas.values()), positive_leave_one=sum(v > 0 for v in values),
                nonpositive_leave_one=sum(v <= 0 for v in values)))
        for name in METHODS:
            selected = [r for r in macros if r['dataset'] == dataset and r['variant'] == name]
            stats = dict(dataset=dataset, variant=name, seeds=len(seeds))
            for metric in METRICS:
                a = np.asarray([r[metric] for r in selected], np.float64)
                stats[metric + '_mean'] = float(a.mean())
                stats[metric + '_sample_sd'] = float(a.std(ddof=1)) if len(a) > 1 else None
            across.append(stats)
        for control in ('A4', 'O1'):
            for metric in METRICS:
                values = []
                for training_seed in seeds:
                    base = next(r for r in macros if (r['dataset'], r['seed'], r['variant']) == (dataset, training_seed, 'A0'))
                    other = next(r for r in macros if (r['dataset'], r['seed'], r['variant']) == (dataset, training_seed, control))
                    values.append(base[metric] - other[metric])
                a = np.asarray(values, np.float64)
                paired.append(dict(dataset=dataset, comparison='A0-minus-' + control, metric=metric,
                    **{'seed' + str(s): (values[seeds.index(s)] if s in seeds else None) for s in SEEDS}, mean_delta_pp=float(a.mean()),
                    sample_sd_delta_pp=float(a.std(ddof=1)) if len(a) > 1 else None,
                    positive_seeds=sum(v > 0 for v in values), negative_seeds=sum(v < 0 for v in values), ties=sum(v == 0 for v in values)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='adnvcnet_public_paired_seed_sensitivity_v1', unit='percent_or_percentage_point_difference',
        category_macros=macros, macro_mean_sample_sd=across, paired_macro_differences=paired,
        category_differences=category_delta, all_leave_one_summaries=leave_one, macro_sensitivity_summary=sensitivity,
        sources=sources or [], new_training=0, new_model_forwards=0, new_AUROC_calculations=0, GT_IO=0,
        partial_analysis=bool(set(lookup) != complete_keys),
        posthoc=True, all_exclusions_reported=True, scope='Descriptive paired training-seed variability and saved-category reaggregation; not leave-one-category-out training, cross-validation, or independent validation', completed_at=now())
    write_json_new(output / 'results.json', result)
    for name, table in (('per_seed_macro.csv', macros), ('macro_mean_sample_sd.csv', across),
                        ('paired_macro_differences.csv', paired), ('all_category_differences.csv', category_delta),
                        ('all_leave_one_summaries.csv', leave_one), ('macro_sensitivity_summary.csv', sensitivity)):
        _csv(output / name, table)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    boot = sub.add_parser('bootstrap', help='Fixed-model paired stratified image-score bootstrap')
    source = boot.add_mutually_exclusive_group(required=True)
    source.add_argument('--scores', type=Path, help='Round-trip image-score CSV, including paper/plot_data/bootstrap_scores.csv')
    source.add_argument('--predictions-root', type=Path, action='append', help='Repeat for disjoint roots; requires completed metrics.json for each method')
    boot.add_argument('--mvtec-root', type=Path)
    boot.add_argument('--visa-root', type=Path)
    boot.add_argument('--training-seed', type=int, default=3407)
    boot.add_argument('--replicates', type=int, default=10000)
    boot.add_argument('--seed', type=int, default=20261028)
    boot.add_argument('--output', type=Path, required=True)
    boot.add_argument('--allow-partial', action='store_true', help='Explicit subset/smoke analysis; never a full-paper reproduction')
    pair = sub.add_parser('paired', help='Paired three-seed deltas and post-hoc leave-one-category macro summaries')
    pair.add_argument('--metrics-root', type=Path, action='append', required=True)
    pair.add_argument('--output', type=Path, required=True)
    pair.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'bootstrap':
        if args.scores:
            cases, sources = load_scores_csv(args.scores), [{'kind': 'saved_image_scores', **_binding(args.scores)}]
            if any(c['training_seed'] != args.training_seed for c in cases):
                parser.error('The scores CSV training seed does not match --training-seed')
        else:
            cases, sources = load_prediction_cases(args.predictions_root,
                {'mvtec': args.mvtec_root, 'visa': args.visa_root}, args.training_seed)
        result = bootstrap(cases, args.output, replicates=args.replicates, seed=args.seed,
                           allow_partial=args.allow_partial, sources=sources)
        print(f'UNCERTAINTY_COMPLETE {len(result["macro"])} macro comparisons {args.output}', flush=True)
    else:
        rows, sources = load_category_metrics(args.metrics_root)
        result = paired_summary(rows, args.output, allow_partial=args.allow_partial, sources=sources)
        print(f'PAIRED_SUMMARY_COMPLETE {len(result["all_leave_one_summaries"])} leave-one rows {args.output}', flush=True)


if __name__ == '__main__':
    main()
