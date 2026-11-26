"""Fit and seal the fixed-total-spatial-weight controls used by Figure 7.

The fit stage reads only the declared normal A images. The predict stage
extends sealed --save-components predictions with center-excluded DINO
retrieval. Run tools.evaluate metrics on the resulting arm directories only
after predict completes; neither stage here reads labels or defect masks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import numpy as np

from models.scoring_controls import RAW_COMPONENTS, baseline, operator
from models.spatial_controls import (ARMS, DEFINITIONS, SPATIAL_TOLERANCE,
    fit_controls, ring_response, score_control)
from tools.ablate import normal_raw_factory
from tools.public_utils import read_json, write_json_new, sha256, now, relative_file, load_profile

RESOURCE_FILES = ('profile.json', 'bank.npz', 'memories.npz', 'calibration.npz')


def resource_identity(resources):
    return {name: sha256(Path(resources)/name) for name in RESOURCE_FILES}


def _profile(resources, runtime):
    profile = load_profile(resources)
    if int(profile['training_seed']) != 3407:
        raise ValueError('The Figure 7 spatial-control protocol uses training seed 3407')
    if profile['dataset']['name'] != runtime.dataset:
        raise ValueError('Resource dataset and runtime differ')
    return profile


def fit_resources(runtime, *, resources, data_root, output, model_identity):
    """Replay normal A raw responses, adding only the eight-offset response."""
    resources, data_root, output = [Path(p).resolve() for p in (resources, data_root, output)]
    profile = _profile(resources, runtime)
    receipt, factory = normal_raw_factory(resources)
    if receipt['dataset'] != runtime.dataset:
        raise ValueError('Calibration dataset and runtime differ')
    if output.exists():
        raise FileExistsError('Spatial fits already exist; choose a fresh output directory')
    output.mkdir(parents=True)
    (output/'normal_s8').mkdir()
    stage = 'normal_S8_retrieval'
    try:
        write_json_new(output/'plan.json', dict(schema='adnvcnet_spatial_control_plan_v1',
            variants=DEFINITIONS, training_seed=3407, S8='Eight valid 3x3 offsets excluding the center',
            normal_A_only=True, normal_B_used=False, CLS_mapping_unchanged=True,
            spatial_replay_atol=SPATIAL_TOLERANCE, GT_masks_read=False, created_at=now()))
        raw_rows, s8_rows, records, errors = [], [], [], []
        for index, (row, raw) in enumerate(zip(receipt['inputs'], factory(), strict=True)):
            path = relative_file(data_root, row['path'])
            if sha256(path) != row['sha256']:
                raise ValueError('Normal A image changed after calibration')
            s8, error = ring_response(runtime, path, raw['spatial_r0'], raw['spatial_r1'],
                                      anchor=index in (0, len(receipt['inputs'])-1))
            target = output/'normal_s8'/f'{index:06d}.npz'
            with target.open('xb') as stream:
                np.savez_compressed(stream, raw_spatial_s8=s8)
            records.append(dict(path=row['path'], owner=row['owner'], sha256=row['sha256'],
                source_raw_sha256=row['raw_sha256'], S8_file=target.relative_to(output).as_posix(),
                S8_sha256=sha256(target)))
            raw_rows.append(raw);s8_rows.append(s8);errors.append(error)
            print(f'SPATIAL_NORMAL {index+1}/{len(receipt["inputs"])}', flush=True)
        stage = 'normal_mapping_fitting'
        calibration = operator(runtime.dataset).load_calibration(resources/'calibration.npz')
        fits, top = fit_controls(raw_rows, s8_rows, calibration, runtime.dataset)
        arrays = {f'{name}__{key}': value for name, fit in fits.items() for key, value in fit.items()}
        arrays.update({f'normal_top__{name}': np.asarray(value) for name, value in top.items()})
        with (output/'fits.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        normal = dict(schema='adnvcnet_spatial_control_normal_v1', dataset=runtime.dataset,
            category=profile['dataset']['category'], training_seed=3407, variants=list(ARMS),
            model_identity=model_identity, resource_sha256=resource_identity(resources),
            calibration_receipt_sha256=sha256(resources/'calibration.json'),
            normal_split_sha256=sha256(resources/'normal_split.json'),
            fits='fits.npz', fits_sha256=sha256(output/'fits.npz'),
            inputs=records, normal_A_count=len(records), normal_A0_Top_fit_exact=True,
            spatial_replay_max=max(errors), normal_only=True, GT_masks_read=False,
            new_CNN_forwards=0, new_DINO_forwards=len(records), completed_at=now())
        write_json_new(output/'normal.json', normal)
        write_json_new(output/'normal_seal.json', dict(schema='adnvcnet_spatial_control_normal_seal_v1',
            manifest='normal.json', manifest_sha256=sha256(output/'normal.json'),
            count=len(records), GT_masks_read=False, sealed_at=now()))
        return normal
    except BaseException:
        write_json_new(output/'failure.json', dict(stage=stage, traceback=traceback.format_exc(),
            partial_outputs_preserved=True, at=now()))
        raise


def load_fits(folder, resources, model_identity):
    folder, resources = Path(folder).resolve(), Path(resources).resolve()
    seal = read_json(folder/'normal_seal.json')
    if seal.get('schema') != 'adnvcnet_spatial_control_normal_seal_v1' or seal.get('GT_masks_read') is not False:
        raise ValueError('Invalid normal spatial-control seal')
    normal = read_json(relative_file(folder, seal['manifest']))
    if sha256(folder/'normal.json') != seal['manifest_sha256'] or len(normal['inputs']) != seal['count']:
        raise ValueError('Spatial normal manifest changed after fitting')
    if normal.get('GT_masks_read') is not False or normal.get('normal_only') is not True:
        raise ValueError('Spatial fitting must use normal A only')
    if normal['resource_sha256'] != resource_identity(resources) or normal['model_identity'] != model_identity:
        raise ValueError('Spatial fits were produced with different resources or models')
    if normal['normal_split_sha256'] != sha256(resources/'normal_split.json') or normal['calibration_receipt_sha256'] != sha256(resources/'calibration.json'):
        raise ValueError('Normal source receipt changed')
    path = relative_file(folder, normal['fits'])
    if sha256(path) != normal['fits_sha256']:
        raise ValueError('Spatial mapping parameters changed')
    for row in normal['inputs']:
        if sha256(relative_file(folder, row['S8_file'])) != row['S8_sha256']:
            raise ValueError('Normal S8 response changed')
    with np.load(path, allow_pickle=False) as data:
        fits = {name: {key.split('__', 1)[1]: data[key].copy()
                      for key in data.files if key.startswith(name+'__')}
                for name in (*ARMS, 'S8')}
    return normal, fits


def predict_controls(runtime, *, resources, data_root, fits_dir, predictions, output, model_identity):
    """Add S8 to sealed raw components; finish all arms before any metrics."""
    from tools.evaluate import verify_predictions
    resources, data_root, predictions, output = [Path(p).resolve() for p in (resources, data_root, predictions, output)]
    profile = _profile(resources, runtime)
    normal, fits = load_fits(fits_dir, resources, model_identity)
    source = verify_predictions(predictions, data_root)
    if source.get('variant') != 'A0' or source.get('save_components') is not True:
        raise ValueError('Use A0 evaluate predict --save-components output')
    if source['dataset'] != runtime.dataset or source['category'] != profile['dataset']['category']:
        raise ValueError('Source predictions and resources identify different categories')
    if source['training_seed'] != 3407 or source['resource_sha256'] != normal['resource_sha256']:
        raise ValueError('Source predictions and spatial fits identify different seed/resources')
    if source['checkpoint_sha256'] != model_identity['checkpoint_sha256']:
        raise ValueError('Source predictions used a different CNN checkpoint')
    if output.exists():
        raise FileExistsError('Spatial predictions already exist; choose a fresh output directory')
    output.mkdir(parents=True)
    (output/'raw_s8').mkdir()
    for arm in ARMS:(output/arm/'predictions').mkdir(parents=True)
    stage = 'spatial_prediction'
    try:
        calibration = operator(runtime.dataset).load_calibration(resources/'calibration.npz')
        rows = {arm: [] for arm in ARMS};s8_records=[];errors=[]
        for index, row in enumerate(source['rows']):
            path = relative_file(data_root, row['input'])
            with np.load(relative_file(predictions, row['prediction']), allow_pickle=False) as data:
                raw = {name: data['raw_'+name].copy() for name in RAW_COMPONENTS+('cls',)}
                old_pixel = data['anomaly_map'].copy();old_score=float(data['image_score'])
            tail = baseline(raw, calibration, runtime.dataset)
            if not np.array_equal(tail['pixel_score'], old_pixel) or abs(float(tail['image_score'])-old_score)>1e-12:
                raise ValueError('A0 replay differs from the source seal')
            s8, error = ring_response(runtime, path, raw['spatial_r0'], raw['spatial_r1'],
                                      anchor=index in (0, source['count']-1))
            errors.append(error)
            target = output/'raw_s8'/f'{index:06d}.npz'
            with target.open('xb') as stream:np.savez_compressed(stream, raw_spatial_s8=s8)
            s8_records.append(dict(input=row['input'], file=target.relative_to(output).as_posix(), sha256=sha256(target)))
            for arm in ARMS:
                result=score_control(raw,s8,calibration,fits,runtime.dataset,arm)
                if arm=='A0' and (not np.array_equal(result['anomaly_map'],old_pixel) or abs(float(result['image_score'])-old_score)>1e-12):
                    raise ValueError('Spatial control A0 changed the original prediction')
                if not all(np.isfinite(value).all() for value in result.values()):
                    raise ValueError('Nonfinite spatial control result')
                target=output/arm/'predictions'/f'{index:06d}.npz'
                with target.open('xb') as stream:np.savez_compressed(stream,**result)
                rows[arm].append(dict(input=row['input'],input_sha256=row['input_sha256'],
                    prediction=target.relative_to(output/arm).as_posix(),prediction_sha256=sha256(target)))
            print(f'SPATIAL_PREDICTED {index+1}/{source["count"]}',flush=True)
        for arm in ARMS:
            folder=output/arm
            manifest={key:value for key,value in source.items() if key not in ('rows','completed_at')}
            manifest.update(rows=rows[arm],count=len(rows[arm]),variant=arm,save_components=False,
                control_definition=DEFINITIONS[arm],spatial_fit_manifest_sha256=sha256(Path(fits_dir)/'normal.json'),
                source_manifest_sha256=sha256(predictions/'predictions.json'),
                source_seal_sha256=sha256(predictions/'prediction_seal.json'),
                model_identity=model_identity,spatial_replay_max=max(errors),new_CNN_forwards=0,
                new_DINO_forwards=source['count'],GT_masks_read=False,completed_at=now())
            write_json_new(folder/'predictions.json',manifest)
            write_json_new(folder/'prediction_seal.json',dict(schema='adnvcnet_public_prediction_seal_v1',
                manifest='predictions.json',manifest_sha256=sha256(folder/'predictions.json'),
                count=len(rows[arm]),GT_masks_read=False,sealed_at=now()))
        write_json_new(output/'complete.json',dict(schema='adnvcnet_spatial_control_predictions_v1',
            variants=list(ARMS),count=source['count'],S8_sources=s8_records,spatial_replay_max=max(errors),
            GT_masks_read=False,metrics_computed=False,completed_at=now()))
        return output
    except BaseException:
        write_json_new(output/'failure.json',dict(stage=stage,traceback=traceback.format_exc(),
            partial_outputs_preserved=True,at=now()))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('fit','predict'):
        current=sub.add_parser(command)
        for option in ('resources','data-root','checkpoint','dino-repo','dino-weights','output'):
            current.add_argument('--'+option,type=Path,required=True)
        current.add_argument('--device',default='cuda')
        if command=='predict':
            current.add_argument('--fits-dir',type=Path,required=True)
            current.add_argument('--predictions',type=Path,required=True)
    args=parser.parse_args(argv)
    from models.public_inference import PublicInference
    profile=load_profile(args.resources)
    runtime=PublicInference(args.checkpoint,profile,args.dino_repo,args.dino_weights,args.resources,args.device)
    identity=dict(checkpoint_sha256=sha256(args.checkpoint),dino_weights_sha256=sha256(args.dino_weights))
    common=dict(resources=args.resources,data_root=args.data_root,output=args.output,model_identity=identity)
    if args.command=='fit':
        value=fit_resources(runtime,**common)
        print(f'SPATIAL_NORMAL_SEALED {value["normal_A_count"]}',flush=True)
    else:
        print(predict_controls(runtime,fits_dir=args.fits_dir,predictions=args.predictions,**common),flush=True)


if __name__=='__main__':main()
