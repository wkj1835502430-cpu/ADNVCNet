"""Train ADNVCNet from normal images using the published category profiles.

Default mode prints a plan. --mode train explicitly starts the60-epoch recipe.
--mode smoke is a labeled one-epoch/limited-batch implementation check.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _state_sha(state, prefix):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key.startswith(prefix):
            h.update(key[len(prefix):].encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _canonical_sha(dataset, category):
    profile = Path(__file__).resolve().parents[1] / 'configs/resource_profiles.json'
    if not profile.is_file():
        return None
    return json.loads(profile.read_text(encoding='utf-8'))['datasets'][dataset][category]['normal_sha256']


def run_category(cfg, args):
    import torch
    from torch.utils.data import DataLoader
    from datasets.public_data import NormalImageDataset, normal_samples
    from models.public_training import (PseudoAnomalyGenerator, initialize_training_model,
                                        optimizer_for, rebuild_training_bank, set_seed, training_step)
    ds, category = cfg['dataset']['name'], cfg['dataset']['category']
    seed = cfg['training_seed']
    if args.data_root is None:
        raise ValueError('--data-root is required for training and smoke modes')
    records = normal_samples(args.data_root, ds, category, args.layout, args.split_file,
                              canonical_sha256=_canonical_sha(ds, category))
    if len(records) != cfg['normal_count']:
        raise ValueError(f'Expected {cfg["normal_count"]} normal images, found {len(records)}')
    if args.mode == 'smoke':
        records = records[:args.smoke_samples]
    dataset = NormalImageDataset(records, image_size=cfg['model']['image_size'])
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = Path(args.output).expanduser().resolve() / ds / category / f'seed{seed}' / 'train'
    save_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    torch.set_num_threads(2)
    model = initialize_training_model(cfg, device=device, imagenet_weights=args.imagenet_weights)
    initial_teacher = _state_sha(model.state_dict(), 'teacher.')
    initial_student = _state_sha(model.state_dict(), 'student.')
    bank_summary = rebuild_training_bank(model, dataset, device)
    frozen_bank = _state_sha(model.state_dict(), 'prototype_bank.')
    optimizer = optimizer_for(model, cfg)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(cfg['train']['batch_size']), shuffle=True,
                        num_workers=0, pin_memory=device.type == 'cuda', drop_last=False,
                        generator=loader_generator)
    generator = PseudoAnomalyGenerator(cfg['pseudo'], category, seed=seed)
    epochs = 1 if args.mode == 'smoke' else int(cfg['train']['epochs'])
    history, best_loss, best_epoch, optimizer_steps = [], float('inf'), -1, 0
    started = time.monotonic()
    resolved = copy.deepcopy(cfg)
    resolved['dataset']['root'] = str(Path(args.data_root).resolve())
    resolved['dataset']['layout'] = args.layout
    _write(save_dir / 'resolved_config.json', resolved)
    _write(save_dir / 'normal_inputs.json', {'normal_only': True, 'test_or_gt_read': False,
           'images': [r.relative_path for r in records], 'sample_count': len(records)})
    for epoch in range(1, epochs + 1):
        model.student.train()
        model.teacher.eval()
        totals, modes, batches = Counter(), Counter(), 0
        for batch_index, images in enumerate(loader):
            if args.mode == 'smoke' and batch_index >= args.smoke_batches:
                break
            values, drawn_modes = training_step(model, images, generator, cfg, optimizer,
                                                 device, epoch, batch_index)
            totals.update(values)
            modes.update(drawn_modes)
            batches += 1
            optimizer_steps += 1
            if (batch_index + 1) % 20 == 0:
                print(f'{ds}/{category} seed={seed} epoch={epoch} batch={batch_index+1} loss={values["train_loss"]:.6f}', flush=True)
        if not batches:
            raise RuntimeError('No training batches produced')
        row = {'epoch': epoch, **{k: v / batches for k, v in totals.items()},
               'batches': batches, 'pseudo_modes': dict(modes)}
        history.append(row)
        _write(save_dir / 'history.json', history)
        payload = {'config': resolved, 'epoch': epoch,
                   'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'train_loss': row['train_loss'], 'training_seed': seed,
                   'selection': 'minimum_training_loss_only', 'smoke': args.mode == 'smoke',
                   'scientific_status': 'Fresh public source run; no historical checkpoint identity or metric-equivalence claim.'}
        torch.save(payload, save_dir / 'last.pth')
        if row['train_loss'] < best_loss:
            best_loss, best_epoch = row['train_loss'], epoch
            torch.save(payload, save_dir / 'best.pth')
        print(f'{ds}/{category} seed={seed} epoch={epoch}/{epochs} loss={row["train_loss"]:.6f}', flush=True)
    assert initial_teacher == _state_sha(model.state_dict(), 'teacher.'), 'Teacher changed'
    assert frozen_bank == _state_sha(model.state_dict(), 'prototype_bank.'), 'Training bank changed'
    result = {'mode': args.mode, 'dataset': ds, 'category': category, 'seed': seed,
              'epochs': epochs, 'optimizer_steps': optimizer_steps, 'normal_images': len(records),
              'best_epoch': best_epoch, 'best_train_loss': best_loss,
              'teacher_unchanged': True, 'training_bank_unchanged': True,
              'initial_teacher_sha256': initial_teacher, 'initial_student_sha256': initial_student,
              'final_student_sha256': _state_sha(model.state_dict(), 'student.'),
              'bank_summary': bank_summary, 'test_or_gt_read': False,
              'elapsed_seconds': time.monotonic() - started,
              'best_checkpoint': str(save_dir / 'best.pth'),
              'full_recipe_completed': args.mode == 'train',
              'accuracy_evaluation_performed': False}
    _write(save_dir / 'training_summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional public YAML with dataset/data_root/category/seed/output')
    parser.add_argument('--dataset', choices=['mvtec', 'visa'])
    parser.add_argument('--category', default=None, help='One category or all')
    parser.add_argument('--seed', type=int, choices=[3407,3408,3409])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--layout', choices=['auto','prepared','raw'], default='auto')
    parser.add_argument('--split-file', type=Path, help='Official VisA1cls.csv for raw-layout input')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--imagenet-weights', type=Path, help='Optional local official torchvision ResNet18 ImageNet state dictionary')
    parser.add_argument('--device', default=None)
    parser.add_argument('--mode', choices=['plan','smoke','train'], default='plan')
    parser.add_argument('--smoke-samples', type=int, choices=[6,12], default=6)
    parser.add_argument('--smoke-batches', type=int, choices=[1,2], default=1)
    args = parser.parse_args(argv)
    supplied = {}
    if args.config:
        import yaml
        supplied = yaml.safe_load(args.config.read_text(encoding='utf-8-sig')) or {}
    for name, fallback in [('dataset', None), ('category','all'), ('seed',3407),
                            ('data_root',None), ('output','outputs/training')]:
        if getattr(args,name) is None:
            setattr(args,name,supplied.get(name,fallback))
    if args.dataset is None:
        parser.error('--dataset or a YAML dataset is required')
    from models.public_training import load_training_profile
    profiles = json.loads((Path(__file__).resolve().parents[1] / 'configs/training_profiles.json').read_text(encoding='utf-8'))
    categories = list(profiles['datasets'][args.dataset]) if args.category == 'all' else [args.category]
    configs = [load_training_profile(args.dataset,c,args.seed) for c in categories]
    if args.mode == 'plan':
        print(json.dumps({'mode':'plan', 'dataset':args.dataset, 'categories':categories,
                          'seed':args.seed, 'epochs':60, 'batch_size':6,
                          'requires': ['normal images','official ImageNet initialization'],
                          'uses_test_or_gt':False, 'runs_started':0}, indent=2))
        return
    for cfg in configs:
        print(json.dumps(run_category(cfg,args),indent=2),flush=True)


if __name__ == '__main__':
    main()
