"""Build normal prototype/DINO resources and fit calibration from normal A.

Example:
  python -m tools.prepare_resources --dataset mvtec --category bottle \
    --data-root /data/mvtec --checkpoint outputs/train/bottle/best.pth \
    --dino-repo /opt/dinov2 --dino-weights /weights/dinov2_vits14.pth
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import traceback

from models.resource_builders import (align_normal_paths, build_balanced_bank,
    build_dino_memories, file_sha256, load_resource_profile, normal_split, save_npz)
from tools.public_utils import now, write_json_new


def prepare_resources(*, dataset, category, data_root, checkpoint, dino_repo,
                      dino_weights, output, device='cuda', seed=3407, layout='auto',
                      split_file=None, resource_profiles=None, skip_calibration=False):
    """Create a new per-category directory; never replace an existing run.

    Inputs are user-provided images and model weights. No historical contract,
    stored prediction, accepted bank, or local machine root is consulted.
    """
    from datasets.public_data import normal_paths
    from models.public_training import load_training_profile
    from models.public_inference import FeatureExtractor, PublicInference

    root = Path(data_root).resolve()
    destination = Path(output).resolve()
    for location in [Path(checkpoint), Path(dino_weights)]:
        if not location.is_file():
            raise FileNotFoundError(location)
    if not (Path(dino_repo) / 'hubconf.py').is_file():
        raise FileNotFoundError('DINOv2 source must contain hubconf.py')
    if destination.exists():
        raise FileExistsError(f'Resource directory already exists: {destination}; choose a new --output')
    resource_profile = load_resource_profile(dataset, category, resource_profiles)
    profile = load_training_profile(dataset, category, seed=seed)
    candidates = normal_paths(root, dataset, category, layout=layout, split_file=split_file)
    paths, hashes = align_normal_paths(candidates, resource_profile)
    split = normal_split(paths, hashes, resource_profile, root)
    protocol = resource_profile['protocol']
    destination.mkdir(parents=True, exist_ok=False)
    current_stage = 'initialization'
    try:
        configuration = copy.deepcopy(profile)
        configuration['resource_build'] = dict(
            schema='adnvcnet_normal_resources_v1', dataset=dataset, category=category,
            created_at=now(), source='constructed_from_user_normal_data',
            protocol=copy.deepcopy(protocol), normal_count=len(paths), reference_count=len(split['reference']),
            calibration_A_count=len(split['A']), calibration_B_count=len(split['B']),
            checkpoint={'file': Path(checkpoint).name, 'sha256': file_sha256(checkpoint)},
            dino_weights={'file': Path(dino_weights).name, 'sha256': file_sha256(dino_weights)},
            device=str(device), claimed_bit_exact_historical_checkpoint=False)
        write_json_new(destination / 'profile.json', configuration)
        write_json_new(destination / 'normal_split.json', split)
        print(f'Normal inputs: {len(paths)}; reference={len(split["reference"])}; A={len(split["A"])}; B={len(split["B"])}', flush=True)
        extractor = FeatureExtractor(checkpoint, profile, dino_repo, dino_weights, device)
        current_stage = 'balanced_prototype_bank'
        print('Building the balanced normal teacher bank', flush=True)
        bank = build_balanced_bank(extractor.teacher_features, paths, hashes,
                                  int(protocol['bank_capacity']), int(protocol['bank_per_image_cap']))
        bank_receipt = save_npz(destination / 'bank.npz', bank)
        del bank
        current_stage = 'dino_normal_memories'
        reference_paths = [root / item['path'] for item in split['reference']]
        print('Building global, spatial and CLS normal memories', flush=True)
        memories = build_dino_memories(extractor.dino_features, reference_paths,
                                      int(resource_profile['memory_seed']),
                                      int(protocol['memory_patches_per_image']),
                                      int(protocol['memory_max_patches']), str(device))
        memory_receipt = save_npz(destination / 'memories.npz', memories)
        del memories, extractor
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json_new(destination / 'reference_resources.json', dict(
            status='COMPLETE_NORMAL_REFERENCE_RESOURCES', bank=bank_receipt,
            memories=memory_receipt, normal_split_sha256=file_sha256(destination / 'normal_split.json'),
            profile_sha256=file_sha256(destination / 'profile.json'), normal_only=True, completed_at=now()))
        calibration_receipt = None
        if not skip_calibration:
            current_stage = 'normal_A_calibration'
            from tools.calibrate import calibrate_resources
            runtime = PublicInference(checkpoint, configuration, dino_repo, dino_weights, destination, device)
            calibration_receipt = calibrate_resources(runtime, root, split, destination)
        write_json_new(destination / 'complete.json', dict(
            status='COMPLETE_REFERENCES_ONLY' if skip_calibration else 'COMPLETE_NORMAL_RESOURCES_AND_CALIBRATION',
            dataset=dataset, category=category, normal_only=True, bank=bank_receipt,
            memories=memory_receipt, calibration=calibration_receipt, completed_at=now()))
        return destination
    except BaseException:
        write_json_new(destination / 'failure.json', dict(stage=current_stage,
            traceback=traceback.format_exc(), original_outputs_preserved=True,
            automatic_retry=False, at=now()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--category', required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dino-repo', required=True, type=Path)
    parser.add_argument('--dino-weights', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='New category resource directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=3407, choices=[3407, 3408, 3409])
    parser.add_argument('--layout', default='auto', choices=['auto', 'prepared', 'raw'])
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--resource-profiles', type=Path)
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Build references only; tools.calibrate can fit normal A later')
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path('outputs/resources') / args.dataset / args.category
    print(prepare_resources(**vars(args)), flush=True)


if __name__ == '__main__':
    main()
