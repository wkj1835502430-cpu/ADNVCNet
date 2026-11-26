"""Meaningful numerical and evidence-order checks for saved-score analyses."""
import csv
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import roc_auc_score

from tests._temporary import test_directory
from tools.public_utils import sha256, write_json_new
from tools.uncertainty import (CASE_INDEX, SCORE_COLUMNS, bootstrap, direct_auc, doubled_wins,
    load_prediction_cases, load_scores_csv, paired_summary, quantile_interval, validate_cases, weights_for)


class UncertaintyTests(unittest.TestCase):
    def test_integer_ties_and_bootstrap_match_explicit_sample_expansion(self):
        rng = np.random.default_rng(7331)
        for n0, n1 in ((1, 1), (2, 3), (17, 11)):
            labels = np.r_[np.zeros(n0, np.uint8), np.ones(n1, np.uint8)]
            weights = weights_for(labels, 37, [123, n0, n1])
            np.testing.assert_array_equal(weights[:, labels == 0].sum(axis=1), np.repeat(n0, 37))
            np.testing.assert_array_equal(weights[:, labels == 1].sum(axis=1), np.repeat(n1, 37))
            for scores in (rng.integers(-2, 3, size=len(labels)).astype(float), np.zeros(len(labels)),
                           labels.astype(float), 1 - labels.astype(float)):
                wins = doubled_wins(scores, labels, weights)
                self.assertAlmostEqual(direct_auc(scores, labels), 100 * roc_auc_score(labels, scores), places=12)
                for j in range(37):
                    indices = np.repeat(np.arange(len(labels)), weights[j])
                    positive = scores[indices][labels[indices] == 1]
                    negative = scores[indices][labels[indices] == 0]
                    brute = sum(2 * (p > n) + (p == n) for p in positive for n in negative)
                    self.assertEqual(int(wins[j]), brute)
                    self.assertEqual(wins[j] / (2 * n0 * n1) * 100,
                                     direct_auc(scores[indices], labels[indices]))

    def test_original_pcg64_draw_order_golden_and_linear_intervals(self):
        labels = np.array([0, 0, 1, 1, 1])
        # Original script: complete normal stratum matrix, then complete anomalous matrix.
        expected = [[1, 1, 1, 1, 1], [1, 1, 0, 2, 1], [2, 0, 0, 3, 0],
                    [1, 1, 2, 0, 1], [1, 1, 2, 0, 1], [1, 1, 2, 1, 0]]
        np.testing.assert_array_equal(weights_for(labels, 6, [20261028, 12]), expected)
        self.assertEqual(quantile_interval(np.arange(5.), .05), [.1, 3.9])
        np.testing.assert_allclose(quantile_interval(np.arange(5.), .05 / 4), [.025, 3.975], rtol=0, atol=1e-15)

    @staticmethod
    def cases():
        scores = np.array([[.2, .2, .2], [.4, .7, .5], [.4, .3, .2], [.9, .9, .9], [.8, .6, .7]])
        labels = np.array([0, 0, 1, 1, 1], np.uint8)
        return [dict(dataset='mvtec', category=c, case_index=CASE_INDEX['mvtec', c], training_seed=3407,
                     inputs=[f'{c}/test/{"good" if y == 0 else "bad"}/{i}.png' for i, y in enumerate(labels)],
                     input_sha256=['a' * 64] * len(labels), scores=scores + k, labels=labels)
                for k, c in enumerate(('bottle', 'cable'))]

    def test_csv_roundtrip_order_identity_and_default_completeness(self):
        cases = self.cases()
        with test_directory() as root:
            path = root / 'scores.csv'
            rows = []
            for c in cases:
                for i, y in enumerate(c['labels']):
                    rows.append(dict(dataset=c['dataset'], category=c['category'], case_index=c['case_index'], image_index=i,
                        input=c['inputs'][i], input_sha256=c['input_sha256'][i], label=int(y),
                        A0=repr(float(c['scores'][i, 0])), A4=repr(float(c['scores'][i, 1])),
                        O1=repr(float(c['scores'][i, 2])), training_seed=3407))
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=SCORE_COLUMNS)
                writer.writeheader()
                writer.writerows(reversed(rows))  # Explicit image_index, not CSV line order, defines pairing.
            actual = validate_cases(load_scores_csv(path), allow_partial=True)
            for a, b in zip(cases, actual):
                np.testing.assert_array_equal(a['scores'], b['scores'])
                np.testing.assert_array_equal(a['labels'], b['labels'])
                self.assertEqual(a['inputs'], b['inputs'])
            with self.assertRaisesRegex(ValueError, 'all 27'):
                validate_cases(actual)
            actual[0]['inputs'][0] = str((Path.cwd() / 'outside' / 'input.png').resolve())
            with self.assertRaisesRegex(ValueError, 'relative'):
                validate_cases(actual, allow_partial=True)

    def test_macro_ci_averages_each_method_then_subtracts_paired_replicates(self):
        cases = self.cases()
        with test_directory() as root:
            result = bootstrap(cases, root / 'bootstrap', replicates=137, allow_partial=True)
            self.assertEqual(result['replicates'], 137)
            self.assertTrue(result['partial_analysis'])
            self.assertEqual(result['family_macro_comparisons'], 4)
            aucs, points = [], []
            for case in cases:
                path = root / 'bootstrap' / ('mvtec_' + case['category'])
                with np.load(path / 'replicates.npz') as saved:
                    aucs.append(saved['I_AUROC'])
                points.append([direct_auc(case['scores'][:, j], case['labels']) for j in range(3)])
            macro = np.mean(np.stack(aucs), axis=0)
            point = np.mean(np.stack(points), axis=0)
            for row, j in zip(result['macro'], (1, 2)):
                self.assertEqual(row['delta_I_AUROC_pp'], float(point[0] - point[j]))
                self.assertEqual([row['ci95_low_pp'], row['ci95_high_pp']], quantile_interval(macro[:, 0] - macro[:, j], .05))
                self.assertEqual([row['familywise95_bonferroni_low_pp'], row['familywise95_bonferroni_high_pp']],
                                 quantile_interval(macro[:, 0] - macro[:, j], .05 / 4))

    @staticmethod
    def prediction_fixture(root):
        data = root / 'data'
        data.mkdir()
        images = []
        for index in range(4):
            target = data / ('image' + str(index) + '.png')
            target.write_bytes(b'identity-only fixture ' + bytes([index]))
            images.append(target)
        folders = []
        for name, scores in (('A0', [0., 2., 1., 3.]), ('A4', [0., 3., 1., 2.]), ('O1', [1., 2., 0., 3.])):
            folder = root / 'predictions' / name
            folder.mkdir(parents=True)
            rows = []
            for index, (image, score) in enumerate(zip(images, scores)):
                target = folder / (str(index) + '.npz')
                np.savez_compressed(target, image_score=score)
                rows.append(dict(input=image.name, input_sha256=sha256(image), prediction=target.name, prediction_sha256=sha256(target)))
            manifest = dict(schema='adnvcnet_public_predictions_v1', dataset='mvtec', category='bottle', training_seed=3407,
                variant=name, count=4, rows=rows, GT_masks_read=False, checkpoint_sha256='b' * 64)
            write_json_new(folder / 'predictions.json', manifest)
            write_json_new(folder / 'prediction_seal.json', dict(schema='adnvcnet_public_prediction_seal_v1', manifest='predictions.json',
                manifest_sha256=sha256(folder / 'predictions.json'), GT_masks_read=False, count=4))
            write_json_new(folder / 'metrics.json', dict(schema='adnvcnet_public_metrics_v1', dataset='mvtec', category='bottle', training_seed=3407,
                variant=name, count=4, unit='percent', image_auroc=direct_auc(np.array(scores), np.array([0, 0, 1, 1])),
                ground_truth=[{'label': y, 'mask_sha256': None} for y in (0, 0, 1, 1)], prediction_seal_sha256=sha256(folder / 'prediction_seal.json')))
            folders.append(folder)
        return data, folders

    def test_all_prediction_seals_verified_before_any_label_consumption(self):
        with test_directory() as root:
            data, folders = self.prediction_fixture(root)
            cases, bindings = load_prediction_cases([root / 'predictions'], {'mvtec': data})
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(bindings), 3)
            np.testing.assert_array_equal(cases[0]['labels'], [0, 0, 1, 1])
            (folders[-1] / '0.npz').write_bytes(b'tampered')
            with patch('tools.uncertainty._labels_from_metrics') as labels:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    load_prediction_cases([root / 'predictions'], {'mvtec': data})
                labels.assert_not_called()

    def test_seed_pairing_sample_sd_and_all_category_exclusions(self):
        rows = []
        for seed in (3407, 3408, 3409):
            k = seed - 3407
            for category, base in (('bottle', 90.), ('cable', 80.), ('capsule', 70.)):
                for name, delta in (('A0', 0.), ('A4', .5 + k), ('O1', 1. + 2 * k)):
                    rows.append(dict(dataset='mvtec', category=category, training_seed=seed, variant=name,
                        image_auroc=base - delta, pixel_auroc=98. if name != 'A4' else 97., aupro=95. if name != 'A4' else 94.))
        with test_directory() as root:
            result = paired_summary(rows, root / 'paired', allow_partial=True)
            paired = next(r for r in result['paired_macro_differences'] if r['comparison'] == 'A0-minus-O1' and r['metric'] == 'image_auroc')
            self.assertEqual([paired['seed3407'], paired['seed3408'], paired['seed3409']], [1., 3., 5.])
            self.assertEqual(paired['mean_delta_pp'], 3.)
            self.assertEqual(paired['sample_sd_delta_pp'], 2.)
            self.assertEqual(len(result['all_leave_one_summaries']), 9)
            self.assertEqual(set(r['excluded_category'] for r in result['all_leave_one_summaries']), {'bottle', 'cable', 'capsule'})
            for row in result['all_leave_one_summaries']:
                self.assertEqual(row['retained_categories'], 2)
                self.assertEqual(row['delta_pp'], 1. + 2 * (row['seed'] - 3407))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                paired_summary(rows + [rows[0]], root / 'duplicate', allow_partial=True)

    def test_invalid_labels_scores_and_case_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            direct_auc(np.array([1., np.nan]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            weights_for(np.array([0, 0]), 10, [20261028, 0])
        cases = self.cases()
        cases[0]['case_index'] = 0  # MVTec starts at 12, even in subset runs.
        with self.assertRaisesRegex(ValueError, 'indexed'):
            validate_cases(cases, allow_partial=True)


if __name__ == '__main__':
    unittest.main()
