"""Summarize user-computed category metrics and repeated training seeds."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS = ('image_auroc', 'pixel_auroc', 'aupro')
CATEGORY_COUNTS = {'mvtec': 15, 'visa': 12}


def summarize(root: Path, output: Path, require_all_categories: bool = False):
    files = sorted(Path(root).rglob('metrics.json'))
    groups = defaultdict(list)
    seen = set()
    for path in files:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if value.get('schema') != 'adnvcnet_public_metrics_v1':
            continue
        dataset = value['dataset'];category = value['category']
        seed = value.get('training_seed');variant = value.get('variant', 'A0')
        if seed not in (3407, 3408, 3409):
            raise ValueError(f'Missing or unsupported training_seed in {path}')
        key = (dataset, seed, variant, category)
        if key in seen:
            raise ValueError(f'Duplicate category metrics for {key}; select a narrower --root')
        seen.add(key)
        if any(value.get(name) is None or not np.isfinite(value[name]) for name in METRICS):
            raise ValueError(f'Undefined or nonfinite metric in {path}')
        groups[(dataset, seed, variant)].append(value)
    if not groups:
        raise ValueError('No public category metrics.json files found')
    macros = []
    for (dataset, seed, variant), values in sorted(groups.items()):
        if require_all_categories and len(values) != CATEGORY_COUNTS[dataset]:
            raise ValueError(f'{dataset}/{seed}/{variant}: expected {CATEGORY_COUNTS[dataset]} categories, found {len(values)}')
        row = {'dataset': dataset, 'training_seed': seed, 'variant': variant,
               'category_count': len(values), 'categories': sorted(v['category'] for v in values),
               **{name: float(np.mean([v[name] for v in values])) for name in METRICS}}
        macros.append(row)
    across = []
    pairs = defaultdict(list)
    for row in macros:pairs[(row['dataset'], row['variant'])].append(row)
    for (dataset, variant), rows in sorted(pairs.items()):
        if any(row['categories'] != rows[0]['categories'] for row in rows):
            raise ValueError(f'Seed comparisons require the same categories: {dataset}/{variant}')
        across.append({'dataset': dataset, 'variant': variant,
                       'seeds': [row['training_seed'] for row in rows], 'category_count': rows[0]['category_count'],
                       'mean': {name: float(np.mean([row[name] for row in rows])) for name in METRICS},
                       'sample_sd': {name: float(np.std([row[name] for row in rows], ddof=1)) if len(rows) > 1 else None for name in METRICS}})
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'adnvcnet_public_seed_summary_v1', 'unit': 'percent',
              'category_aggregation': 'equal-weight mean of unrounded category metrics',
              'seed_standard_deviation': 'sample SD, denominator n-1; null for a single seed',
              'category_macros': macros, 'seed_summary': across}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with (output / 'category_macros.csv').open('w', encoding='utf-8', newline='') as stream:
        names = ['dataset', 'training_seed', 'variant', 'category_count', *METRICS]
        writer = csv.DictWriter(stream, fieldnames=names);writer.writeheader()
        writer.writerows({key: row[key] for key in names} for row in macros)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-all-categories', action='store_true')
    args = parser.parse_args(argv)
    result = summarize(args.root, args.output, args.require_all_categories)
    print(f'SUMMARY_COMPLETE {len(result["category_macros"])} category-macro groups')


if __name__ == '__main__':
    main()
