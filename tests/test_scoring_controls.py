"""Normal-only synthetic checks for fixed scoring interventions."""
import unittest
import numpy as np
from models.scoring_controls import (ALL_VARIANTS, baseline, control_pixel,
    fit_arrays, fit_controls, load_fit_arrays, operator, quantile_map, score_control,
    separate_top)


def raw_fixture(seed, shift=0.):
    rng = np.random.default_rng(seed)
    return dict(cnn=rng.uniform(0., 2., (512, 512)).astype(np.float32) + shift,
                **{k: rng.uniform(.01, .4, (37, 37)) + shift for k in ('global','spatial_r0','spatial_r1')},
                cls=np.asarray(.1 + .01 * seed))


def fixture_calibration(dataset, rows):
    op = operator(dataset)
    cal = {name: op.fit(np.concatenate([op.sample(row[name]) for row in rows])) for name in op.COMPONENTS}
    cal['cls'] = op.fit([float(row['cls']) for row in rows])
    source = lambda r: dict(raw_global=r['global'], raw_spatial_r0=r['spatial_r0'], raw_spatial_r1=r['spatial_r1'])
    cal['top'] = op.fit([op.pixel(row['cnn'], source(row), cal)['pixel_top1'] for row in rows])
    return cal


class ScoringControlTests(unittest.TestCase):
    def test_quantile_mapping_keeps_negative_and_upper_extrapolation(self):
        np.testing.assert_array_equal(quantile_map(np.asarray([0., 1., 3., 5.]), [1., 3.]), [-.05, 0., .1, .2])
        with self.assertRaises(ValueError):quantile_map(np.asarray([1.]), [1., 1.])

    def test_shared_pixel_maps_and_removed_image_terms(self):
        rows = [raw_fixture(i) for i in range(3)]
        for dataset in ('mvtec', 'visa'):
            cal = fixture_calibration(dataset, rows)
            fits = fit_controls(lambda: iter(rows), cal, dataset, ALL_VARIANTS)
            raw = raw_fixture(7, .03);tail = baseline(raw, cal, dataset)
            output = {v: score_control(raw, cal, dataset, v, fits, tail) for v in ALL_VARIANTS}
            for variant in ('A0','A5','A6','O1','TAIL_PIXEL','ID_TOP','ID_CLS'):
                np.testing.assert_array_equal(output[variant]['anomaly_map'], tail['pixel_score'])
            self.assertEqual(float(output['A5']['image_score']), float(tail['image_top_evidence']))
            self.assertEqual(float(output['A6']['image_score']), float(tail['image_cls_evidence']))
            self.assertEqual(float(output['A0']['image_score']), float(tail['image_score']))
            self.assertEqual(float(output['ID_TOP']['image_score']), .5*float(tail['pixel_top1'])+.5*float(tail['image_cls_evidence']))
            self.assertEqual(float(output['ID_CLS']['image_score']), .5*float(tail['image_top_evidence'])+.5*float(raw['cls']))
            # Removing CNN pixel evidence makes the map invariant to a change
            # confined to the CNN raw response; other calibrations remain fixed.
            altered = dict(raw, cnn=raw['cnn'] + 9.)
            np.testing.assert_array_equal(output['A1']['anomaly_map'], score_control(altered, cal, dataset, 'A1', fits)['anomaly_map'])
            packed = fit_arrays(fits)
            restored = load_fit_arrays(packed, ALL_VARIANTS, 3)
            for variant in ALL_VARIANTS:
                np.testing.assert_array_equal(output[variant]['image_score'], score_control(raw, cal, dataset, variant, restored)['image_score'])

    def test_changed_A_inputs_cannot_reuse_an_old_top_calibration(self):
        rows = [raw_fixture(i) for i in range(3)]
        cal = fixture_calibration('mvtec', rows)
        with self.assertRaises(ValueError):
            fit_controls(lambda: iter([raw_fixture(8)]), cal, 'mvtec')

    def test_single_identity_replacement_changes_only_the_selected_plane(self):
        rows = [raw_fixture(i) for i in range(3)]
        cal = fixture_calibration('mvtec', rows)
        raw = raw_fixture(7);tail = baseline(raw, cal, 'mvtec')
        # A change confined to CNN cannot alter a control's DINO planes; the
        # CNN identity response enters with exactly the retained one-half weight.
        altered = dict(raw, cnn=raw['cnn'] + np.float32(2.))
        from models.scoring_controls import control_pixel
        before = control_pixel(raw, tail, 'ID_CNN')
        after = control_pixel(altered, tail, 'ID_CNN')
        np.testing.assert_allclose(after-before, 1., rtol=0, atol=1.2e-7)

    def test_independent_top_cannot_be_smaller_before_mapping(self):
        rows = [raw_fixture(i) for i in range(3)]
        for dataset in ('mvtec','visa'):
            cal = fixture_calibration(dataset, rows)
            tail = baseline(raw_fixture(10), cal, dataset)
            self.assertGreaterEqual(separate_top(tail, dataset) + 1e-12, operator(dataset).top_fraction(tail['pixel_score']))

    def test_T1_and_DINO_G_do_not_fit_or_use_any_calibration(self):
        raw = dict(cnn=np.ones((512,512), np.float32), **{k:np.full((37,37),v) for k,v in [('global',2.),('spatial_r0',3.),('spatial_r1',4.)]}, cls=np.asarray(.25))
        for dataset in ('mvtec','visa'):
            def forbidden_normal_factory():
                raise AssertionError('Identity/global-only controls must not fit normal mappings')
            fits = fit_controls(forbidden_normal_factory, None, dataset, ('T1','DINO_G'))
            self.assertFalse(fits['top_fits'])
            t1 = score_control(raw, None, dataset, 'T1', fits)
            dg = score_control(raw, None, dataset, 'DINO_G', fits)
            np.testing.assert_array_equal(t1['anomaly_map'], np.full((512,512),2.))
            self.assertEqual(float(t1['image_score']),1.125)
            np.testing.assert_array_equal(dg['anomaly_map'],np.full((512,512),2.))
            self.assertEqual(float(dg['image_score']),2.)


if __name__ == '__main__':unittest.main()
