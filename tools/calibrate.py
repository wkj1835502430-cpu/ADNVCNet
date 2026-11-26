"""Fit all six response mappings using normal calibration subset A only.

Run as ``python -m tools.calibrate --help`` from the repository root.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from models.public_inference import PublicInference
from tools.public_utils import read_json, write_json_new, sha256, now, relative_file, load_profile


def calibrate_resources(runtime, data_root: Path, normal_split: dict, output: Path) -> dict:
    """Fit local, Top and CLS mappings from the same ordered normal A subset.

    ``owner`` identifies a training image's prototypes, which are excluded from
    CNN matching for calibration on both datasets. DINO reference and A subsets
    must be disjoint. B is never used to fit or select a mapping.
    """
    data_root = Path(data_root).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'calibration.npz'
    if target.exists() or (output / 'calibration.json').exists():
        raise FileExistsError('Calibration already exists; use a fresh resources directory')
    rows = normal_split.get('A', [])
    if not rows:
        raise ValueError('normal_split.json has no normal calibration subset A')
    reference = {r['path'] for r in normal_split.get('reference', [])}
    normal = {r['path']: r for r in normal_split['normal_inputs']}
    if reference.intersection(r['path'] for r in rows):
        raise ValueError('Normal references overlap calibration subset A')
    raw_dir = output / 'calibration_raw'
    raw_dir.mkdir(exist_ok=False)
    operator = runtime.operator
    pools = {name: [] for name in operator.COMPONENTS}
    cls_values = []
    records = []
    for index, row in enumerate(rows):
        if row['path'] not in normal or normal[row['path']]['owner'] != row['owner']:
            raise ValueError('Calibration owner does not match normal input ordering')
        path = relative_file(data_root, row['path'])
        digest = sha256(path)
        if row.get('sha256') and digest != row['sha256']:
            raise ValueError('Normal image changed after resource construction')
        raw = runtime.raw_components(path, exclude_index=int(row['owner']))
        raw_path = raw_dir / f'{index:06d}.npz'
        with raw_path.open('xb') as stream:
            np.savez_compressed(stream, **raw)
        for name in operator.COMPONENTS:
            pools[name].append(operator.sample(raw[name]))
        cls_values.append(float(raw['cls']))
        records.append({'path': row['path'], 'owner': int(row['owner']), 'sha256': digest,
                        'raw_file': raw_path.relative_to(output).as_posix(), 'raw_sha256': sha256(raw_path)})
        print(f'CALIBRATION_RAW {index + 1}/{len(rows)} {row["path"]}', flush=True)
    calibration = {name: operator.fit(np.concatenate(pools[name])) for name in operator.COMPONENTS}
    top_values = []
    for row in records:
        with np.load(output / row['raw_file'], allow_pickle=False) as data:
            raw = {name: data[name] for name in data.files}
            score = operator.pixel(raw['cnn'], runtime.dino_source(raw), calibration)
            top_values.append(float(score['pixel_top1']))
    calibration['top'] = operator.fit(top_values)
    calibration['cls'] = operator.fit(cls_values)
    with target.open('xb') as stream:
        np.savez_compressed(stream, **{name + '__' + key: value for name, part in calibration.items() for key, value in part.items()})
    runtime.calibration = calibration
    receipt = {'schema': 'adnvcnet_public_normal_calibration_v1', 'dataset': runtime.dataset,
               'normal_A_count': len(rows), 'normal_only': True, 'GT_masks_read': False,
               'normal_B_used_for_fitting': False, 'cnn_leave_one_image_out': True,
               'local_sampling': 'CNN at most 4096 fixed positions; each DINO 37x37 map retained in full',
               'mappings': ['cnn', 'global', 'spatial_r0', 'spatial_r1', 'top', 'cls'],
               'top_and_cls_use_same_A_images_as_local_mappings': True,
               'file': target.name, 'sha256': sha256(target), 'inputs': records,
               'run_identity': 'new_user_generated_resources', 'completed_at': now()}
    write_json_new(output / 'calibration.json', receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--resources', type=Path, required=True)
    result.add_argument('--data-root', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--dino-repo', type=Path, required=True)
    result.add_argument('--dino-weights', type=Path, required=True)
    result.add_argument('--device', default='cuda')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    profile = load_profile(args.resources)
    split = read_json(args.resources / 'normal_split.json')
    runtime = PublicInference(args.checkpoint, profile, args.dino_repo, args.dino_weights, args.resources, args.device)
    receipt = calibrate_resources(runtime, args.data_root, split, args.resources)
    print(f'CALIBRATION_COMPLETE {receipt["normal_A_count"]} normal A images', flush=True)


if __name__ == '__main__':
    main()
