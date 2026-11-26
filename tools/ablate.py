"""Derive fixed scoring controls from saved normal A and sealed raw components.

First run tools.prepare_resources (or tools.calibrate), then
tools.evaluate predict --save-components. This command runs no model forward
and never opens a test mask. Each variant gets a standard prediction seal;
tools.evaluate metrics can evaluate it in a separate command.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import traceback
import numpy as np

from models.scoring_controls import (ALL_VARIANTS, DEFAULT_VARIANTS, DESCRIPTIONS,
    NO_FIT_VARIANTS, RAW_COMPONENTS, baseline, fit_arrays, fit_controls, operator, score_control)
from tools.public_utils import read_json, relative_file, sha256, write_json_new, now


def normal_raw_factory(resources):
    resources = Path(resources).resolve()
    receipt = read_json(resources / 'calibration.json')
    split = read_json(resources / 'normal_split.json')
    if receipt.get('normal_only') is not True or receipt.get('normal_B_used_for_fitting') is not False:
        raise ValueError('Calibration must use only declared normal subset A')
    if sha256(resources / 'calibration.npz') != receipt['sha256']:
        raise ValueError('Calibration changed after its normal receipt')
    expected = [(x['path'], int(x['owner']), x.get('sha256')) for x in split['A']]
    actual = [(x['path'], int(x['owner']), x.get('sha256')) for x in receipt['inputs']]
    if actual != expected:
        raise ValueError('Normal A owner/path ordering differs from the resource split')
    if len({x[0] for x in expected}) != len(expected):
        raise ValueError('Duplicate normal A path in calibration receipt')
    def factory():
        for row in receipt['inputs']:
            path = relative_file(resources, row['raw_file'])
            if sha256(path) != row['raw_sha256']:
                raise ValueError('Stored normal raw components changed')
            with np.load(path, allow_pickle=False) as data:
                raw = {key: data[key].copy() for key in RAW_COMPONENTS + ('cls',)}
            if not all(np.isfinite(v).all() for v in raw.values()):
                raise ValueError('Nonfinite normal raw component')
            yield raw
    return receipt, factory


def derive(*, resources, predictions, data_root, output, variants=DEFAULT_VARIANTS):
    from tools.evaluate import verify_predictions
    resources, predictions, output = map(lambda p: Path(p).resolve(), (resources, predictions, output))
    variants = tuple(dict.fromkeys(variants))
    if not variants or any(v not in ALL_VARIANTS for v in variants):
        raise ValueError('Choose fixed variants from ' + ', '.join(ALL_VARIANTS))
    if output.exists():
        raise FileExistsError('Control output exists; choose a new directory')
    normal_receipt, raw_factory = normal_raw_factory(resources)
    dataset = normal_receipt['dataset']
    op = operator(dataset)
    calibration = op.load_calibration(resources / 'calibration.npz')
    output.mkdir(parents=True, exist_ok=False)
    current_stage = 'normal_A_fitting'
    try:
        write_json_new(output / 'control_plan.json', dict(
            schema='adnvcnet_fixed_score_control_plan_v1', variants=list(variants),
            definitions={v: DESCRIPTIONS[v] for v in variants},
            pixel_linear_percentiles=[90., 99.5], pixel_linear_targets=[0., .1],
            pixel_linear_quantile_method='linear', pixel_linear_clipping=False,
            calibration_sha256=sha256(resources / 'calibration.npz'),
            normal_receipt_sha256=sha256(resources / 'calibration.json'),
            normal_A_owners=[r['owner'] for r in normal_receipt['inputs']],
            test_masks_read=False, created_at=now()))
        fits = fit_controls(raw_factory, calibration, dataset, variants)
        with (output / 'control_fits.npz').open('xb') as stream:
            np.savez_compressed(stream, **fit_arrays(fits))
        write_json_new(output / 'normal_fit.json', dict(
            variants=list(variants), normal_A_count=fits['normal_A_count'],
            original_top_reconstructed_exact=fits['original_top_reconstructed_exact'],
            no_mapping_fit_variants=[v for v in variants if v in NO_FIT_VARIANTS],
            fits_sha256=sha256(output / 'control_fits.npz'),
            normal_only=True, GT_masks_read=False, completed_at=now()))
        current_stage = 'sealed_component_derivation'
        source = verify_predictions(predictions, data_root)
        if source['dataset'] != dataset or source.get('save_components') is not True:
            raise ValueError('Use matching evaluate predict --save-components output')
        for name in ('profile.json', 'bank.npz', 'memories.npz', 'calibration.npz'):
            if source['resource_sha256'][name] != sha256(resources / name):
                raise ValueError(f'Source predictions use different resources: {name}')
        rows = {v: [] for v in variants}
        for variant in variants:
            (output / variant / 'predictions').mkdir(parents=True)
        for index, item in enumerate(source['rows']):
            path = relative_file(predictions, item['prediction'])
            if sha256(path) != item['prediction_sha256']:
                raise ValueError('Source prediction changed')
            with np.load(path, allow_pickle=False) as saved:
                try:
                    raw = {key: saved['raw_' + key].copy() for key in RAW_COMPONENTS + ('cls',)}
                except KeyError as exc:
                    raise ValueError('Missing raw components; rerun predict with --save-components') from exc
                tail = baseline(raw, calibration, dataset)
                if not np.array_equal(tail['pixel_score'], saved['anomaly_map']):
                    raise ValueError('A0 pixel replay differs from the sealed prediction')
                if abs(float(tail['image_score']) - float(saved['image_score'])) > 1e-12:
                    raise ValueError('A0 image-score replay differs from the sealed prediction')
            shared_map = tail['pixel_score']
            for variant in variants:
                result = score_control(raw, calibration, dataset, variant, fits, tail)
                if variant in ('A0', 'A5', 'A6', 'O1', 'TAIL_PIXEL', 'ID_TOP', 'ID_CLS') and not np.array_equal(result['anomaly_map'], shared_map):
                    raise ValueError('A shared-map control changed the pixel map')
                if not all(np.isfinite(v).all() for v in result.values()):
                    raise ValueError('Nonfinite control output')
                target = output / variant / 'predictions' / f'{index:06d}.npz'
                with target.open('xb') as stream:
                    np.savez_compressed(stream, **result)
                rows[variant].append(dict(input=item['input'], input_sha256=item['input_sha256'],
                    prediction=target.relative_to(output / variant).as_posix(), prediction_sha256=sha256(target)))
            print(f'CONTROLS_DERIVED {index + 1}/{source["count"]}', flush=True)
        for variant in variants:
            folder = output / variant
            manifest = {key: value for key, value in source.items() if key not in ('rows', 'completed_at')}
            manifest.update(rows=rows[variant], count=len(rows[variant]), save_components=False,
                variant=variant, control_variant=variant, control_definition=DESCRIPTIONS[variant],
                control_fits_sha256=sha256(output / 'control_fits.npz'),
                source_manifest_sha256=sha256(predictions / 'predictions.json'),
                source_seal_sha256=sha256(predictions / 'prediction_seal.json'),
                run_identity='new_user_generated_fixed_score_controls', GT_masks_read=False,
                new_model_forwards=0, completed_at=now())
            write_json_new(folder / 'predictions.json', manifest)
            write_json_new(folder / 'prediction_seal.json', dict(
                schema='adnvcnet_public_prediction_seal_v1', manifest='predictions.json',
                manifest_sha256=sha256(folder / 'predictions.json'), count=len(rows[variant]),
                GT_masks_read=False, sealed_at=now()))
        write_json_new(output / 'complete.json', dict(
            status='COMPLETE_FIXED_SCORE_CONTROL_PREDICTIONS', variants=list(variants),
            count=source['count'], normal_A_count=fits['normal_A_count'],
            new_model_forwards=0, GT_masks_read=False, metrics_computed=False, completed_at=now()))
        return output
    except BaseException:
        write_json_new(output / 'failure.json', dict(stage=current_stage,
            traceback=traceback.format_exc(), partial_outputs_preserved=True,
            automatic_retry=False, at=now()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='Sealed evaluate predict --save-components directory')
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variants', nargs='+', choices=ALL_VARIANTS + ('all',), default=DEFAULT_VARIANTS)
    args = parser.parse_args(argv)
    if 'all' in args.variants:
        if list(args.variants) != ['all']:
            parser.error('Use --variants all alone, or list individual variants')
        args.variants = ALL_VARIANTS
    print(derive(**vars(args)), flush=True)


if __name__ == '__main__':
    main()
