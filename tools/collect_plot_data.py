"""Collect complete, newly computed statistics into paper_figures CSV schemas.

No model, mask, parameter fitting, or metric calculation is involved. Each
requested figure must have all 15 MVTec AD and 12 VisA categories.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import numpy as np

from datasets.public_data import MVTEC_CATEGORIES, VISA_CATEGORIES
from tools.public_utils import now, read_json, sha256, write_json_new

METRICS = ('image_auroc', 'pixel_auroc', 'aupro')
CATEGORIES = {'mvtec': MVTEC_CATEGORIES, 'visa': VISA_CATEGORIES}
CONTROLS = ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'T1')
SPATIAL = ('A0', 'S1_DOUBLE', 'S0_S8')
STAGES = ('preprocess', 'host_to_device', 'cnn_feature_response', 'dino_feature',
          'matching_raw', 'tail', 'control_and_sync')


def _csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source(path, **metadata):
    path = Path(path)
    return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size, **metadata)


def _finite(value, description, *, nonnegative=False, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Missing/non-numeric ' + description) from exc
    if not math.isfinite(number) or (nonnegative and number < 0) or (positive and number <= 0):
        raise ValueError('Invalid ' + description)
    return number


def load_metrics(roots, variants, training_seed):
    lookup, sources, seen_paths = {}, [], set()
    for root_index, root in enumerate(roots):
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob('metrics.json')):
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            row = read_json(path)
            if row.get('schema') != 'adnvcnet_public_metrics_v1' or row.get('variant') not in variants:
                continue
            if 'training_seed' not in row:
                raise ValueError('Missing training_seed in ' + str(path))
            if row['training_seed'] != training_seed:
                continue
            key = row['dataset'], row['category'], row['variant']
            if key[0] not in CATEGORIES or key[1] not in CATEGORIES[key[0]]:
                raise ValueError('Unknown dataset/category in ' + str(path))
            if row.get('unit') != 'percent' or not isinstance(row.get('count'), int) or row['count'] < 1:
                raise ValueError('Expected evaluated category metrics in percent with a positive image count')
            for metric in METRICS:
                value = _finite(row.get(metric), metric)
                if not 0 <= value <= 100:
                    raise ValueError('Category metrics must be percentages between zero and 100')
            duplicate = key in lookup
            if duplicate:
                previous = lookup[key]
                if key[2] != 'A0' or any(previous[k] != row[k] for k in (*METRICS, 'count', 'unit', 'training_seed')):
                    raise ValueError('Conflicting or duplicate category/variant metrics: ' + str(key))
            else:
                lookup[key] = row
            sources.append(_source(path, root_index=root_index, relative_file=path.relative_to(root).as_posix(),
                dataset=key[0], category=key[1], variant=key[2], deduplicated_identical_A0=duplicate))
    return lookup, sources


def _complete(lookup, variants):
    expected = {(dataset, category, variant) for dataset, categories in CATEGORIES.items()
                for category in categories for variant in variants}
    missing = sorted(expected - set(lookup))
    if missing:
        raise ValueError('Incomplete figure inputs: all 15 MVTec AD and 12 VisA categories are required; '
                         f'missing {len(missing)} category/variant rows, starting with {missing[:3]}')
    for dataset, categories in CATEGORIES.items():
        for category in categories:
            if len({lookup[dataset, category, variant]['count'] for variant in variants}) != 1:
                raise ValueError('Paired variants have different image counts: ' + dataset + '/' + category)


def ablation_rows(lookup):
    _complete(lookup, ('A0', *CONTROLS))
    rows = []
    for dataset, categories in CATEGORIES.items():
        macros = {name: {metric: float(np.mean([lookup[dataset, category, name][metric]
                    for category in categories])) for metric in METRICS} for name in ('A0', *CONTROLS)}
        for control in CONTROLS:
            for metric in METRICS:
                shared = control in ('A5', 'A6') and metric in ('pixel_auroc', 'aupro')
                if shared and any(lookup[dataset, category, 'A0'][metric] != lookup[dataset, category, control][metric]
                                  for category in categories):
                    raise ValueError('A5/A6 must preserve the A0 pixel metrics')
                rows.append(dict(dataset=dataset, variant=control, metric=metric,
                    A0_minus_control_pp=macros['A0'][metric] - macros[control][metric],
                    inherited_identical_pixel_map=shared))
    return rows


def order_rows(lookup):
    _complete(lookup, ('A0', 'O1'))
    rows, means = [], []
    for dataset in ('visa', 'mvtec'):
        deltas = []
        for category in CATEGORIES[dataset]:
            base, control = lookup[dataset, category, 'A0'], lookup[dataset, category, 'O1']
            if any(base[k] != control[k] for k in ('pixel_auroc', 'aupro')):
                raise ValueError('A0/O1 must share identical localization metrics')
            value = base['image_auroc'] - control['image_auroc']
            rows.append(dict(dataset=dataset, category=category, joint_minus_separate_pp=value))
            deltas.append(value)
        means.append(dict(dataset=dataset, category='Macro mean', joint_minus_separate_pp=float(np.mean(deltas))))
    return rows + means


def spatial_rows(lookup):
    _complete(lookup, SPATIAL)
    return [dict(dataset=dataset, variant=name, categories=len(categories),
                 **{metric: float(np.mean([lookup[dataset, category, name][metric]
                         for category in categories])) for metric in METRICS})
            for dataset, categories in CATEGORIES.items() for name in SPATIAL]


def validate_intervals(folder, training_seed):
    folder = Path(folder)
    path = folder / 'uncertainty_macro.csv'
    result = read_json(folder / 'results.json')
    if (result.get('schema') != 'adnvcnet_public_uncertainty_v1' or result.get('partial_analysis') is not False
            or result.get('replicates') != 10000 or result.get('seed') != 20261028
            or result.get('training_seed') != training_seed or result.get('family_macro_comparisons') != 4):
        raise ValueError('Fig. 8 requires the complete 10,000-resample public uncertainty analysis')
    rows = _csv_rows(path)
    keys = {(r['dataset'], r['comparison']) for r in rows}
    if len(rows) != 4 or keys != {(ds, comp) for ds in CATEGORIES for comp in ('A0-A4', 'A0-O1')}:
        raise ValueError('Fig. 8 needs the four dataset/contrast interval rows')
    values = ('delta_I_AUROC_pp', 'ci95_low_pp', 'ci95_high_pp',
              'familywise95_bonferroni_low_pp', 'familywise95_bonferroni_high_pp')
    for row in rows:
        if int(row['categories']) != len(CATEGORIES[row['dataset']]):
            raise ValueError('Partial category macro in uncertainty inputs')
        delta, low, high, adjusted_low, adjusted_high = [_finite(row[k], k) for k in values]
        if not adjusted_low <= low <= high <= adjusted_high:
            raise ValueError('Inconsistent ordinary/adjusted interval nesting')
        saved = next(r for r in result['macro'] if (r['dataset'], r['comparison']) == (row['dataset'], row['comparison']))
        if any(float(row[k]) != saved[k] for k in values):
            raise ValueError('Uncertainty CSV differs from its unrounded results.json')
    return path, [_source(path), _source(folder / 'results.json')]


def validate_profile(folder, training_seed):
    folder = Path(folder)
    paths = [folder / 'efficiency_classes.csv', folder / 'efficiency_summary.json']
    rows, summary = _csv_rows(paths[0]), read_json(paths[1])
    keys = [(r['dataset'], r['category']) for r in rows]
    expected = {(ds, c) for ds, categories in CATEGORIES.items() for c in categories}
    if len(keys) != 27 or len(set(keys)) != 27 or set(keys) != expected:
        raise ValueError('Fig. 11 requires exactly all 27 profiled categories; smoke/subset runs cannot be plotted as complete')
    if (summary.get('schema') != 'adnvcnet_public_profile_summary_v1' or summary.get('mode') != 'run'
            or summary.get('training_seed') != training_seed or summary.get('batch_size') != 1
            or summary.get('stage_keys') != list(STAGES)):
        raise ValueError('Expected a completed synchronized batch-one public profile run')
    if set(summary['datasets']) != set(CATEGORIES):
        raise ValueError('Both dataset profile summaries are required')
    for row in rows:
        if int(row['measurements']) < 1:
            raise ValueError('Profile class has no measurements')
        total = _finite(row['total_mean_ms'], 'class latency', positive=True)
        stages = [_finite(row[stage + '_mean_ms'], stage, nonnegative=True) for stage in STAGES]
        if abs(sum(stages) - total) > 1e-8:
            raise ValueError('Profile stage means do not sum to class latency')
        for name in ('peak_allocated_GiB', 'peak_reserved_GiB', 'loading_validation_calibration_ms'):
            _finite(row[name], name, nonnegative=True)
    for dataset, categories in CATEGORIES.items():
        group = [r for r in rows if r['dataset'] == dataset]
        item = summary['datasets'][dataset]
        if item['categories'] != len(categories):
            raise ValueError('Profile summary category count mismatch')
        mean = _finite(item['latency_ms']['mean'], 'macro latency', positive=True)
        if abs(mean - float(np.mean([float(r['total_mean_ms']) for r in group]))) > 1e-8:
            raise ValueError('Profile summary is not the equal-category class mean')
        stages = [_finite(item['stage_mean_ms'][stage], stage, nonnegative=True) for stage in STAGES]
        if abs(sum(stages) - mean) > 1e-8:
            raise ValueError('Profile stage means do not sum to dataset latency')
        for name in ('peak_allocated_GiB', 'peak_reserved_GiB'):
            peak = _finite(item[name], name, nonnegative=True)
            if peak != max(float(r[name]) for r in group):
                raise ValueError('Dataset GPU memory peak differs from class maxima')
        _finite(item['loading_validation_calibration_ms']['median'], 'loading median', nonnegative=True)
    return paths, [_source(path) for path in paths]


def collect_plot_data(*, output_data, metrics_roots=(), spatial_roots=(), uncertainty_dir=None,
                      profile_dir=None, training_seed=3407, figures=None):
    output_data = Path(output_data)
    if output_data.exists():
        raise FileExistsError('Select a fresh --output-data directory')
    inferred = ([5, 6] if metrics_roots else []) + ([7] if spatial_roots else [])
    inferred += ([8] if uncertainty_dir else []) + ([11] if profile_dir else [])
    selected = list(figures) if figures is not None else inferred
    if not selected or len(set(selected)) != len(selected) or any(n not in (5, 6, 7, 8, 11) for n in selected):
        raise ValueError('Select figures 5, 6, 7, 8, and/or 11 without duplicates')
    tables, copies, sources = {}, {}, {}
    main = {}
    if set(selected) & {5, 6}:
        if not metrics_roots:
            raise ValueError('Figures 5/6 require --metrics-root')
        needed = {'A0'} | (set(CONTROLS) if 5 in selected else set()) | ({'O1'} if 6 in selected else set())
        main, sources['metrics'] = load_metrics(metrics_roots, needed, training_seed)
        if 5 in selected:
            tables['figure05_ablation.csv'] = ablation_rows(main)
        if 6 in selected:
            tables['figure06_fusion_order.csv'] = order_rows(main)
    if 7 in selected:
        if not spatial_roots:
            raise ValueError('Figure 7 requires --spatial-root')
        spatial, sources['spatial'] = load_metrics(spatial_roots, SPATIAL, training_seed)
        for key in set(main) & set(spatial):
            if any(main[key][k] != spatial[key][k] for k in (*METRICS, 'count')):
                raise ValueError('Spatial and main A0 baselines disagree: ' + str(key))
        tables['figure07_spatial_macro.csv'] = spatial_rows(spatial)
    if 8 in selected:
        if uncertainty_dir is None:
            raise ValueError('Figure 8 requires --uncertainty-dir')
        path, sources['uncertainty'] = validate_intervals(uncertainty_dir, training_seed)
        copies['figure08_intervals.csv'] = path
    if 11 in selected:
        if profile_dir is None:
            raise ValueError('Figure 11 requires --profile-dir')
        paths, sources['profile'] = validate_profile(profile_dir, training_seed)
        copies.update(dict(zip(('figure11_efficiency_classes.csv', 'figure11_efficiency_summary.json'), paths)))
    # Validate every requested complete input before creating any output files.
    output_data.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        _write_csv(output_data / name, rows)
    for name, path in copies.items():
        shutil.copyfile(path, output_data / name)
    result = dict(schema='adnvcnet_collected_plot_data_v1', training_seed=training_seed,
        figures=selected, sources=sources, output_files={name: _source(output_data / name)
            for name in (*tables, *copies)}, unrounded_category_metrics=True, category_aggregation='equal category weights',
        new_training=0, new_model_forwards=0, new_metric_calculation=0, mask_reads=0, collected_at=now())
    write_json_new(output_data / 'collection_manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-root', type=Path, action='append', default=[])
    parser.add_argument('--spatial-root', type=Path, action='append', default=[])
    parser.add_argument('--uncertainty-dir', type=Path)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--training-seed', type=int, default=3407)
    parser.add_argument('--figures', type=int, nargs='+', choices=(5, 6, 7, 8, 11))
    parser.add_argument('--output-data', type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect_plot_data(output_data=args.output_data, metrics_roots=args.metrics_root,
        spatial_roots=args.spatial_root, uncertainty_dir=args.uncertainty_dir, profile_dir=args.profile_dir,
        training_seed=args.training_seed, figures=args.figures)
    print('PLOT_DATA_COLLECTED ' + ','.join(map(str, result['figures'])) + ' ' + str(args.output_data), flush=True)


if __name__ == '__main__':
    main()
