"""Draw figures from a user's sealed predictions and computed metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def metric_plot(args):
    rows = [json.loads(path.read_text(encoding='utf-8-sig')) for path in args.metrics]
    if not rows:
        raise ValueError('Supply at least one category metrics.json')
    identities = [(row['dataset'], row['category']) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Plot one seed/run at a time; duplicate categories were supplied')
    args.output.mkdir(parents=True, exist_ok=False)
    plt = plotting()
    names = [f"{row['dataset']}/{row['category']}" for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(3, len(rows) * .28)), sharey=True)
    for axis, key, label in zip(axes, ('image_auroc', 'pixel_auroc', 'aupro'), ('I-AUROC (%)', 'P-AUROC (%)', 'AUPRO (%)')):
        values = np.asarray([row[key] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Cannot plot undefined or nonfinite metrics')
        axis.barh(np.arange(len(rows)), values, color='#2c6178', height=.65)
        axis.set_xlabel(label)
        axis.set_xlim(0, 100)
        axis.grid(axis='x', color='#dce2e5', linewidth=.5)
        axis.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(rows)), names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.output / 'category_metrics.png', dpi=600, bbox_inches='tight')
    fig.savefig(args.output / 'category_metrics.svg', bbox_inches='tight')
    plt.close(fig)
    (args.output / 'source_metrics.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


def qualitative_plot(args):
    # Importing this module does not instantiate a model or run inference.
    from tools.evaluate import verify_predictions, ground_truth_rows, MaskSequence
    from tools.public_utils import relative_file
    manifest = verify_predictions(args.predictions, args.data_root)
    indices = args.indices if args.indices else list(range(min(4, len(manifest['rows']))))
    if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= len(manifest['rows']) for i in indices):
        raise ValueError('Indices must be distinct valid prediction indices')
    if not np.isfinite(args.vmax) or args.vmax <= 0:
        raise ValueError('--vmax must be positive')
    masks = MaskSequence(ground_truth_rows(args.data_root, manifest, args.split_file)) if args.include_masks else None
    args.output.mkdir(parents=True, exist_ok=False)
    plt = plotting()
    columns = 4 if masks is not None else 3
    fig, axes = plt.subplots(len(indices), columns, figsize=(2.3 * columns, 2.35 * len(indices)), squeeze=False)
    selected = []
    for row_number, index in enumerate(indices):
        row = manifest['rows'][index]
        with Image.open(relative_file(args.data_root, row['input'])) as source:
            image = source.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
        with np.load(args.predictions / row['prediction'], allow_pickle=False) as data:
            score = data['anomaly_map'].copy()
        if score.shape != (512, 512) or not np.isfinite(score).all():
            raise ValueError('Expected a finite 512x512 anomaly map')
        axes[row_number, 0].imshow(image)
        heat = axes[row_number, 1].imshow(score, cmap='viridis', vmin=0, vmax=args.vmax)
        axes[row_number, 2].imshow(image)
        axes[row_number, 2].imshow(score, cmap='viridis', vmin=0, vmax=args.vmax, alpha=.45)
        if masks is not None:
            axes[row_number, 3].imshow(masks[index], cmap='gray', vmin=0, vmax=1)
        for axis in axes[row_number]:
            axis.set_xticks([]);axis.set_yticks([])
            for spine in axis.spines.values():spine.set_visible(False)
        axes[row_number, 0].set_ylabel(f'Image {index}', fontsize=9)
        selected.append({'index': index, 'input': row['input'], 'prediction': row['prediction']})
    for axis, label in zip(axes[0], ['Input', 'Anomaly map', 'Overlay', 'Ground truth']):axis.set_title(label)
    fig.subplots_adjust(left=.07, right=.90, bottom=.025, top=.93, wspace=.12, hspace=.12)
    color_axis=fig.add_axes([.92,.20,.015,.58])
    fig.colorbar(heat,cax=color_axis,label='Calibrated response')
    fig.savefig(args.output / 'qualitative.png', dpi=600, bbox_inches='tight')
    fig.savefig(args.output / 'qualitative.svg', bbox_inches='tight')
    plt.close(fig)
    (args.output / 'figure_sources.json').write_text(json.dumps({
        'prediction_seal_verified': True, 'indices': selected,
        'display_range': [0, args.vmax], 'clipping': 'display only; saved scores are unchanged',
        'GT_masks_shown': masks is not None}, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    metrics = commands.add_parser('metrics')
    metrics.add_argument('--metrics', nargs='+', type=Path, required=True)
    metrics.add_argument('--output', type=Path, required=True)
    qualitative = commands.add_parser('qualitative')
    qualitative.add_argument('--predictions', type=Path, required=True)
    qualitative.add_argument('--data-root', type=Path, required=True)
    qualitative.add_argument('--output', type=Path, required=True)
    qualitative.add_argument('--indices', type=int, nargs='+')
    qualitative.add_argument('--include-masks', action='store_true')
    qualitative.add_argument('--split-file', type=Path)
    qualitative.add_argument('--vmax', type=float, default=10.)
    args = parser.parse_args(argv)
    {'metrics': metric_plot, 'qualitative': qualitative_plot}[args.kind](args)


if __name__ == '__main__':
    main()
