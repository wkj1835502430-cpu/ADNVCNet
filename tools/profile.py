"""Synchronized batch-one public-API timing, without reading test annotations.

Run ``python -m tools.profile --help`` from the repository root.  Plan mode
discovers and hashes inputs but does not instantiate networks or run inference.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import gc
import os
import platform
import time
from pathlib import Path

for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np

from datasets.public_data import MVTEC_CATEGORIES, VISA_CATEGORIES, normal_paths
from tools.evaluate import discover_images
from tools.public_utils import load_profile, now, read_json, relative_file, sha256, write_json_new

PHASES = ('preprocess', 'host_to_device', 'cnn_feature_response', 'dino_feature',
          'matching_raw', 'tail', 'control_and_sync')
GPU_PHASES = frozenset(('host_to_device', 'cnn_feature_response', 'dino_feature', 'matching_raw'))


class Clock:
    """Partition nested host-wall intervals, retaining explicit GPU barriers."""
    def __init__(self, synchronize, clock=time.perf_counter_ns):
        self.sync, self.clock = synchronize, clock
        self.active, self.stack, self.events, self.sync_calls = False, [], [], 0

    def reset(self):
        if self.stack:
            raise RuntimeError('Cannot reset an active timing span')
        self.events, self.sync_calls, self.active = [], 0, True

    @contextlib.contextmanager
    def span(self, label):
        if not self.active:
            yield
            return
        if label != 'total' and label not in PHASES:
            raise ValueError(label)
        gpu = label in GPU_PHASES
        if gpu:
            self.sync()
            self.sync_calls += 1
        frame = dict(label=label, start_ns=self.clock(), children_ns=0, depth=len(self.stack))
        self.stack.append(frame)
        try:
            yield
        finally:
            if gpu:
                self.sync()
                self.sync_calls += 1
            end = self.clock()
            if self.stack.pop() is not frame:
                raise RuntimeError('Unbalanced timing spans')
            frame.update(end_ns=end, inclusive_ns=end - frame['start_ns'])
            frame['exclusive_ns'] = frame['inclusive_ns'] - frame['children_ns']
            if frame['exclusive_ns'] < 0:
                raise RuntimeError('Negative exclusive timing')
            if self.stack:
                self.stack[-1]['children_ns'] += frame['inclusive_ns']
            self.events.append(frame)

    def wrap(self, function, label):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with self.span(label):
                return function(*args, **kwargs)
        return call

    def finish(self, total_ns):
        self.active = False
        roots = [event for event in self.events if event['label'] == 'total']
        if self.stack or len(roots) != 1:
            raise RuntimeError('Expected one completed total span')
        bins, counts = dict.fromkeys(PHASES, 0), dict.fromkeys(PHASES, 0)
        for event in self.events:
            key = 'control_and_sync' if event['label'] == 'total' else event['label']
            bins[key] += event['exclusive_ns']
            counts[key] += 1
        bins['control_and_sync'] += int(total_ns) - roots[0]['inclusive_ns']
        if min(bins.values()) < 0 or sum(bins.values()) != int(total_ns):
            raise RuntimeError('Stage intervals do not partition total latency')
        return dict(total_ns=int(total_ns), exclusive_wall_ns=bins, phase_calls=counts,
                    synchronization_calls=self.sync_calls + 2)


class Instrument:
    """Temporarily wrap existing operations; restore all attributes on exit."""
    def __init__(self, runtime, clock):
        self.runtime, self.clock, self.saved = runtime, clock, []

    def patch(self, owner, name, replacement):
        own = name in vars(owner)
        self.saved.append((owner, name, own, vars(owner).get(name)))
        setattr(owner, name, replacement)

    def wrap(self, owner, name, label):
        self.patch(owner, name, self.clock.wrap(getattr(owner, name), label))

    def __enter__(self):
        import torch
        from models import public_inference as api
        try:
            self.wrap(self.runtime, 'cnn_raw', 'cnn_feature_response')
            self.wrap(self.runtime, 'dino_features', 'dino_feature')
            self.wrap(self.runtime, 'score_raw', 'tail')
            self.wrap(self.runtime.model.prototype_bank, 'distance_map', 'matching_raw')
            self.wrap(api, '_rgb512', 'preprocess')
            self.wrap(self.runtime.extractor, '_tensor', 'preprocess')
            for name in ('_raw_global', '_raw_spatial', 'raw_image_distance'):
                self.wrap(api.dino_math, name, 'matching_raw')
            original_to = torch.Tensor.to
            clock = self.clock

            def transfer(tensor, *args, **kwargs):
                target = kwargs.get('device', args[0] if args else None)
                if isinstance(target, torch.Tensor):
                    target = target.device
                try:
                    is_cuda = target is not None and torch.device(target).type == 'cuda'
                except (TypeError, RuntimeError):
                    is_cuda = False
                if tensor.device.type == 'cpu' and is_cuda:
                    with clock.span('host_to_device'):
                        return original_to(tensor, *args, **kwargs)
                return original_to(tensor, *args, **kwargs)

            self.patch(torch.Tensor, 'to', transfer)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        for owner, name, own, raw in reversed(self.saved):
            if own:
                setattr(owner, name, raw)
            else:
                delattr(owner, name)
        self.saved.clear()


def selected_inputs(root, dataset, category, input_split='test', layout='auto', split_file=None,
                    measurements=30, warmup=10):
    """First paths in deterministic file-path order; never open a test mask."""
    root = Path(root).resolve()
    if input_split == 'test':
        paths, actual_layout = discover_images(root, dataset, category, layout, split_file)
    else:
        paths = sorted(path.relative_to(root).as_posix() for path in
                       normal_paths(root, dataset, category, layout=layout, split_file=split_file))
        actual_layout = layout
    if not (0 <= warmup <= measurements) or measurements < 1:
        raise ValueError('Require 0 <= warmup <= measurements and measurements >= 1')
    if len(paths) < measurements:
        raise ValueError(f'{dataset}/{category}: need {measurements} {input_split} images, found {len(paths)}')
    rows = [{'input': relative, 'sha256': sha256(relative_file(root, relative))}
            for relative in paths[:measurements]]
    return dict(input_split=input_split, layout=actual_layout, measurements=rows, warmup=rows[:warmup],
                selection='first paths in sorted file-path order; warmup overlaps measurement inputs',
                GT_masks_read=False)


def timed_call(runtime, path, expected_sha, clock, synchronize, *, measure=True):
    """Input verification, decoding and inference are inside the timed call."""
    def predict_verified():
        with clock.span('preprocess'):
            if sha256(path) != expected_sha:
                raise ValueError(f'Input changed after planning: {path.name}')
        return runtime.predict(path)
    if not measure:
        return predict_verified(), None
    synchronize()
    clock.reset()
    start = clock.clock()
    try:
        with clock.span('total'):
            result = predict_verified()
            synchronize()
        end = clock.clock()
        return result, clock.finish(end - start)
    finally:
        clock.active = False


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=float(values.min()), maximum=float(values.max()))


def verify_resource_files(resources, checkpoint, dino_weights):
    files = {name: dict(bytes=(resources / name).stat().st_size, sha256=sha256(resources / name))
             for name in ('profile.json', 'bank.npz', 'memories.npz', 'calibration.npz')}
    for name, path in (('checkpoint', checkpoint), ('dino_weights', dino_weights)):
        files[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    complete = resources / 'complete.json'
    if complete.is_file():
        metadata = read_json(complete)
        for name, value in metadata.items():
            if isinstance(value, dict) and value.get('file') in files and value.get('sha256'):
                if files[value['file']]['sha256'] != value['sha256']:
                    raise ValueError(f'Resource seal mismatch: {value["file"]}')
    return files


def run_category(args, job, output):
    import torch
    from models.public_inference import PublicInference, dataset_name
    root, checkpoint, resources = (Path(job[key]) for key in ('data_root', 'checkpoint', 'resources'))
    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        torch.cuda.init()
        synchronize = lambda: torch.cuda.synchronize(args.device)
        synchronize()
        torch.cuda.reset_peak_memory_stats(args.device)
    else:
        synchronize = lambda: None
    start = time.perf_counter_ns()
    bindings = verify_resource_files(resources, checkpoint, args.dino_weights)
    profile = load_profile(resources)
    if dataset_name(profile) != job['dataset']:
        raise ValueError('Resource profile dataset mismatch')
    if int(profile.get('training_seed', 3407)) != args.seed:
        raise ValueError('Resource training seed mismatch')
    runtime = PublicInference(checkpoint, profile, args.dino_repo, args.dino_weights, resources, args.device)
    if runtime.calibration is None:
        raise FileNotFoundError('Fit calibration.npz before profiling')
    synchronize()
    load_ns = time.perf_counter_ns() - start
    clock = Clock(synchronize)
    # Load verification uses normal training images only and is not measured.
    normal = normal_paths(root, job['dataset'], job['category'], layout=args.layout, split_file=args.split_file)[:2]
    verified = []
    for path in normal:
        reference = runtime.predict(path)
        with Instrument(runtime, clock):
            instrumented = runtime.predict(path)
        if set(reference) != set(instrumented) or any(not np.array_equal(reference[key], instrumented[key]) for key in reference):
            raise RuntimeError('Instrumented and uninstrumented normal predictions differ')
        verified.append(dict(input=path.relative_to(root).as_posix(), sha256=sha256(path), exact_output_match=True))
    rows, timings, memory = [], [], []
    with Instrument(runtime, clock):
        for row in job['plan']['warmup']:
            timed_call(runtime, relative_file(root, row['input']), row['sha256'], clock, synchronize, measure=False)
        synchronize()
        for index, row in enumerate(job['plan']['measurements']):
            if cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            _, timing = timed_call(runtime, relative_file(root, row['input']), row['sha256'], clock, synchronize)
            allocated = int(torch.cuda.max_memory_allocated(args.device)) if cuda else 0
            reserved = int(torch.cuda.max_memory_reserved(args.device)) if cuda else 0
            timings.append([timing['total_ns']] + [timing['exclusive_wall_ns'][phase] for phase in PHASES])
            memory.append([allocated, reserved])
            rows.append(dict(index=index, **row, **timing, allocated_bytes=allocated, reserved_bytes=reserved))
            print(f'PROFILED {job["dataset"]}/{job["category"]} {index + 1}/{len(job["plan"]["measurements"])}', flush=True)
    array, mem = np.asarray(timings, dtype=np.int64), np.asarray(memory, dtype=np.int64)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'timings.npz', timing_ns=array, peak_memory_bytes=mem,
                        columns=np.asarray(('total',) + PHASES))
    latency = distribution(array[:, 0] / 1e6)
    summary = dict(dataset=job['dataset'], category=job['category'], measurements=len(rows),
                   input_split=args.input_split, warmup=len(job['plan']['warmup']),
                   total_mean_ms=latency['mean'], total_median_ms=latency['median'], total_p95_ms=latency['p95'],
                   inverse_mean_latency_FPS=1000.0 / latency['mean'],
                   stage_mean_ms={phase: float(array[:, index + 1].mean() / 1e6) for index, phase in enumerate(PHASES)},
                   peak_allocated_GiB=float(mem[:, 0].max() / 2**30) if cuda else None,
                   peak_reserved_GiB=float(mem[:, 1].max() / 2**30) if cuda else None,
                   loading_validation_calibration_ms=load_ns / 1e6,
                   normal_load_verification=verified, resource_files=bindings, rows=rows, GT_masks_read=False)
    write_json_new(output / 'measured.json', summary)
    del runtime
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return summary


def summarize(classes):
    datasets = {}
    for dataset in sorted({item['dataset'] for item in classes}):
        selected = [item for item in classes if item['dataset'] == dataset]
        pooled = [row['total_ns'] / 1e6 for item in selected for row in item['rows']]
        latency = distribution(pooled)
        latency['mean'] = float(np.mean([item['total_mean_ms'] for item in selected]))
        stages = {phase: float(np.mean([item['stage_mean_ms'][phase] for item in selected])) for phase in PHASES}
        datasets[dataset] = dict(categories=len(selected), samples=len(pooled), latency_ms=latency,
                                inverse_mean_latency_FPS=1000 / latency['mean'], stage_mean_ms=stages,
                                stage_fraction={phase: value / latency['mean'] for phase, value in stages.items()},
                                peak_allocated_GiB=max((item['peak_allocated_GiB'] for item in selected if item['peak_allocated_GiB'] is not None), default=None),
                                peak_reserved_GiB=max((item['peak_reserved_GiB'] for item in selected if item['peak_reserved_GiB'] is not None), default=None),
                                loading_validation_calibration_ms=distribution([item['loading_validation_calibration_ms'] for item in selected]))
    return datasets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'run', 'smoke'), default='plan')
    parser.add_argument('--dataset', choices=('mvtec', 'visa', 'all'), required=True)
    parser.add_argument('--category', default='all')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--input-split', choices=('test', 'train'), default='test')
    parser.add_argument('--layout', choices=('auto', 'prepared', 'raw'), default='auto')
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--mvtec-root', type=Path)
    parser.add_argument('--visa-root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoints-root', type=Path, default=Path('outputs/training'))
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--resources-root', type=Path, default=Path('outputs/resources'))
    parser.add_argument('--dino-repo', type=Path, default=Path('third_party/dinov2'))
    parser.add_argument('--dino-weights', type=Path, default=Path('pretrained/dinov2_vits14_pretrain.pth'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.dataset == 'all' and args.category != 'all':
        parser.error('--dataset all requires --category all')
    if args.mode == 'smoke' and (args.category == 'all' or args.dataset == 'all'):
        parser.error('Smoke mode accepts exactly one category')
    if args.output.exists():
        raise FileExistsError('Select a fresh output directory')
    jobs = []
    for dataset in (('mvtec', 'visa') if args.dataset == 'all' else (args.dataset,)):
        categories = MVTEC_CATEGORIES if dataset == 'mvtec' else VISA_CATEGORIES
        if args.category != 'all':
            if args.category not in categories:
                parser.error(f'Unknown {dataset} category: {args.category}')
            categories = (args.category,)
        root = getattr(args, dataset + '_root') or args.data_root
        if root is None:
            parser.error(f'Provide --{dataset}-root or --data-root')
        for category in categories:
            single = len(categories) == 1 and args.dataset != 'all'
            checkpoint = args.checkpoint if single and args.checkpoint else args.checkpoints_root / dataset / category / f'seed{args.seed}' / 'train/best.pth'
            resources = args.resources if single and args.resources else args.resources_root / dataset / category / f'seed{args.seed}'
            plan = selected_inputs(root, dataset, category, args.input_split, args.layout, args.split_file,
                                   measurements=2 if args.mode == 'smoke' else 30, warmup=1 if args.mode == 'smoke' else 10)
            jobs.append(dict(dataset=dataset, category=category, data_root=str(root.resolve()),
                             checkpoint=str(checkpoint.resolve()), resources=str(resources.resolve()), plan=plan))
    args.output.mkdir(parents=True)
    write_json_new(args.output / 'plan.json', dict(schema='adnvcnet_public_profile_plan_v1', mode=args.mode,
                   training_seed=args.seed, batch_size=1, jobs=jobs, GT_masks_read=False, created_at=now()))
    if args.mode == 'plan':
        print(f'PLANNED {len(jobs)} categories; no inference: {args.output}', flush=True)
        return
    import torch
    torch.set_num_threads(1)
    classes = [run_category(args, job, args.output / (job['dataset'] + '_' + job['category'])) for job in jobs]
    fields = ['dataset', 'category', 'measurements', 'total_mean_ms', 'total_median_ms', 'total_p95_ms',
              'inverse_mean_latency_FPS', 'peak_allocated_GiB', 'peak_reserved_GiB', 'loading_validation_calibration_ms']
    fields += [phase + '_mean_ms' for phase in PHASES]
    with (args.output / 'efficiency_classes.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in classes:
            row = {key: item[key] for key in fields if key in item}
            row.update({phase + '_mean_ms': item['stage_mean_ms'][phase] for phase in PHASES})
            writer.writerow(row)
    write_json_new(args.output / 'efficiency_summary.json', dict(schema='adnvcnet_public_profile_summary_v1',
        run_identity='new_public_api_measurements', mode=args.mode, input_split=args.input_split,
        datasets=summarize(classes), stage_keys=list(PHASES), batch_size=1, training_seed=args.seed,
        warmup_per_class=1 if args.mode == 'smoke' else 10, measured_per_class=2 if args.mode == 'smoke' else 30,
        total_scope='Input SHA verification, decoding, transfer, frozen feature extraction, retrieval, calibration and score/map CPU copies of PublicInference.predict; explicit synchronization retained',
        initialization_scope='Resource hashes, model/resource loading, validation and calibration loading; excludes imports, CUDA context creation and two normal numerical-verification forwards',
        memory_scope='PyTorch allocated/reserved peaks including resident tensors; CPU memory not measured',
        FPS_scope='Reciprocal of mean synchronized batch-one latency; not pipelined throughput',
        environment=dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                         cuda=torch.version.cuda, device=args.device,
                         gpu=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else None,
                         cpu_threads=torch.get_num_threads()), GT_masks_read=False, completed_at=now()))
    print(f'PROFILE_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
