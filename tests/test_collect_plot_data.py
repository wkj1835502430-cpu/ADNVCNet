"""Synthetic integration checks for complete numerical paper-figure inputs."""
import csv
import json
import unittest
from pathlib import Path
import numpy as np

from tests._temporary import test_directory
from tools.collect_plot_data import (CATEGORIES, CONTROLS, METRICS, SPATIAL, STAGES,
    collect_plot_data, validate_profile)
from tools.public_utils import read_json, write_json_new


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PlotDataTests(unittest.TestCase):
    @staticmethod
    def metric(dataset, category, variant, index):
        shift = {'A0': 0., 'O1': .2, 'S1_DOUBLE': -.03, 'S0_S8': .04}.get(variant, .1)
        shared = variant in ('A0', 'O1', 'A5', 'A6')
        return dict(schema='adnvcnet_public_metrics_v1', dataset=dataset, category=category,
            training_seed=3407, variant=variant, count=100, unit='percent',
            image_auroc=90. + index / 10 - shift, pixel_auroc=98. if shared else 97.9,
            aupro=95. if shared else 94.9)

    @classmethod
    def metrics_fixture(cls, root):
        for dataset, categories in CATEGORIES.items():
            for index, category in enumerate(categories):
                for name in ('A0', *CONTROLS, 'O1'):
                    target = root / 'controls' / dataset / category / name / 'metrics.json'
                    write_json_new(target, cls.metric(dataset, category, name, index))
                write_json_new(root / 'main' / dataset / category / 'metrics.json', cls.metric(dataset, category, 'A0', index))
                for name in SPATIAL:
                    write_json_new(root / 'spatial' / dataset / category / name / 'metrics.json', cls.metric(dataset, category, name, index))

    @staticmethod
    def intervals_fixture(root):
        root.mkdir()
        rows = [dict(dataset=dataset, comparison=comp, categories=len(categories), delta_I_AUROC_pp=.2,
                     ci95_low_pp=.1, ci95_high_pp=.3, familywise95_bonferroni_low_pp=.05,
                     familywise95_bonferroni_high_pp=.35, adjusted_interval_excludes_zero=True)
                for dataset, categories in CATEGORIES.items() for comp in ('A0-A4', 'A0-O1')]
        csv_write(root / 'uncertainty_macro.csv', rows)
        write_json_new(root / 'results.json', dict(schema='adnvcnet_public_uncertainty_v1', macro=rows,
            replicates=10000, seed=20261028, training_seed=3407, family_macro_comparisons=4, partial_analysis=False))

    @staticmethod
    def profile_fixture(root):
        root.mkdir()
        rows, datasets = [], {}
        for dataset, categories in CATEGORIES.items():
            group = []
            for i, category in enumerate(categories):
                total = 10. + i
                row = dict(dataset=dataset, category=category, measurements=30, total_mean_ms=total,
                    total_median_ms=total, total_p95_ms=total + 1, inverse_mean_latency_FPS=1000 / total,
                    peak_allocated_GiB=.5 + i / 100, peak_reserved_GiB=1. + i / 100,
                    loading_validation_calibration_ms=100. + i,
                    **{s + '_mean_ms': total - 6 if s == 'preprocess' else 1. for s in STAGES})
                rows.append(row)
                group.append(row)
            datasets[dataset] = dict(categories=len(categories), latency_ms={'mean': float(np.mean([r['total_mean_ms'] for r in group]))},
                stage_mean_ms={s: float(np.mean([r[s + '_mean_ms'] for r in group])) for s in STAGES},
                peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in group),
                peak_reserved_GiB=max(r['peak_reserved_GiB'] for r in group), loading_validation_calibration_ms={'median': 106.})
        csv_write(root / 'efficiency_classes.csv', rows)
        write_json_new(root / 'efficiency_summary.json', dict(schema='adnvcnet_public_profile_summary_v1', mode='run',
            training_seed=3407, batch_size=1, stage_keys=list(STAGES), datasets=datasets))

    def test_all_requested_schemas_complete_unrounded_and_copied_exact(self):
        with test_directory() as root:
            self.metrics_fixture(root)
            self.intervals_fixture(root / 'uncertainty')
            self.profile_fixture(root / 'profile')
            result = collect_plot_data(output_data=root / 'data', metrics_roots=[root / 'main', root / 'controls'],
                spatial_roots=[root / 'spatial'], uncertainty_dir=root / 'uncertainty', profile_dir=root / 'profile')
            self.assertEqual(result['figures'], [5, 6, 7, 8, 11])
            expected = {'figure05_ablation.csv': 42, 'figure06_fusion_order.csv': 29, 'figure07_spatial_macro.csv': 6}
            for filename, count in expected.items():
                with (root / 'data' / filename).open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                if filename.startswith('figure05'):
                    shared = [r for r in rows if r['inherited_identical_pixel_map'] == 'True']
                    self.assertEqual(len(shared), 8)
                    self.assertTrue(all(float(r['A0_minus_control_pp']) == 0 for r in shared))
                if filename.startswith('figure06'):
                    for ds, categories in CATEGORIES.items():
                        group = [r for r in rows if r['dataset'] == ds]
                        self.assertEqual(len(group), len(categories) + 1)
                        mean = next(r for r in group if r['category'] == 'Macro mean')
                        self.assertEqual(float(mean['joint_minus_separate_pp']), np.mean([float(r['joint_minus_separate_pp']) for r in group if r['category'] != 'Macro mean']))
            self.assertEqual((root / 'data/figure08_intervals.csv').read_bytes(), (root / 'uncertainty/uncertainty_macro.csv').read_bytes())
            self.assertEqual((root / 'data/figure11_efficiency_classes.csv').read_bytes(), (root / 'profile/efficiency_classes.csv').read_bytes())
            self.assertEqual((root / 'data/figure11_efficiency_summary.json').read_bytes(), (root / 'profile/efficiency_summary.json').read_bytes())
            self.assertEqual(sum(r['deduplicated_identical_A0'] for r in result['sources']['metrics']), 27)

    def test_missing_category_fails_before_creating_output(self):
        with test_directory() as root:
            self.metrics_fixture(root)
            (root / 'controls/visa/candle/A1/metrics.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                collect_plot_data(output_data=root / 'data', metrics_roots=[root / 'controls'])
            self.assertFalse((root / 'data').exists())

    def test_conflicting_duplicate_a0_and_changed_shared_map_are_rejected(self):
        with test_directory() as root:
            self.metrics_fixture(root)
            path = root / 'main/mvtec/bottle/metrics.json'
            value = read_json(path)
            value['image_auroc'] += .01
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                collect_plot_data(output_data=root / 'conflict', metrics_roots=[root / 'main', root / 'controls'])
            path = root / 'controls/mvtec/bottle/A5/metrics.json'
            value = read_json(path)
            value['pixel_auroc'] -= .01
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'preserve'):
                collect_plot_data(output_data=root / 'shared', metrics_roots=[root / 'controls'], figures=[5])

    def test_explicit_order_only_requires_no_unrelated_controls(self):
        with test_directory() as root:
            for dataset, categories in CATEGORIES.items():
                for index, category in enumerate(categories):
                    for name in ('A0', 'O1'):
                        write_json_new(root / 'metrics' / dataset / category / name / 'metrics.json', self.metric(dataset, category, name, index))
            result = collect_plot_data(output_data=root / 'data', metrics_roots=[root / 'metrics'], figures=[6])
            self.assertEqual(result['figures'], [6])
            self.assertEqual(set(result['output_files']), {'figure06_fusion_order.csv'})

    def test_uncertainty_partial_or_changed_csv_is_rejected(self):
        with test_directory() as root:
            self.intervals_fixture(root / 'uncertainty')
            path = root / 'uncertainty/results.json'
            value = read_json(path)
            value['partial_analysis'] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'complete'):
                collect_plot_data(output_data=root / 'data', uncertainty_dir=root / 'uncertainty')
            value['partial_analysis'] = False
            value['macro'][0]['delta_I_AUROC_pp'] = .21
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'differs'):
                collect_plot_data(output_data=root / 'data', uncertainty_dir=root / 'uncertainty')

    def test_profile_incomplete_or_inconsistent_timing_is_rejected(self):
        with test_directory() as root:
            self.profile_fixture(root / 'profile')
            path = root / 'profile/efficiency_classes.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            csv_write(path, rows[:-1])
            with self.assertRaisesRegex(ValueError, 'exactly all 27'):
                validate_profile(root / 'profile', 3407)
            rows[0]['preprocess_mean_ms'] = '0.0'
            csv_write(path, rows)
            with self.assertRaisesRegex(ValueError, 'sum'):
                validate_profile(root / 'profile', 3407)


if __name__ == '__main__':
    unittest.main()
