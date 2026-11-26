"""Normal-only resource construction using the published, fixed sampling rules.

This module has no dependency on historical checkpoints, result directories, or
machine-specific contracts. Feature extraction is supplied by public_inference.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

LAYERS = {'layer1': (128, 128), 'layer2': (64, 64), 'layer3': (32, 32)}
CHANNELS = {'layer1': 64, 'layer2': 128, 'layer3': 256}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_resource_profile(dataset: str, category: str, configuration: str | Path | None = None) -> dict:
    location = Path(configuration) if configuration else Path(__file__).resolve().parents[1] / 'configs/resource_profiles.json'
    data = json.loads(location.read_text(encoding='utf-8'))
    if data.get('schema') != 'adnvcnet_portable_resource_profiles_v1':
        raise ValueError('Unsupported resource-profile schema')
    try:
        result = dict(data['datasets'][dataset][category])
    except KeyError as exc:
        raise ValueError(f'No fixed resource profile for {dataset}/{category}') from exc
    result.update(dataset=dataset, category=category, protocol=dict(data['protocol']))
    return result


def balanced_quotas(normal_count: int, capacity: int = 24576, per_image_cap: int = 512) -> list[int]:
    if normal_count < 1 or capacity < 1 or per_image_cap < 1:
        raise ValueError('Positive normal count, bank capacity and per-image cap are required')
    total = min(capacity, normal_count * per_image_cap)
    quotient, remainder = divmod(total, normal_count)
    return [quotient + (index < remainder) for index in range(normal_count)]


def balanced_positions(input_sha256: str, layer: str, spatial: int, take: int) -> np.ndarray:
    """Exact accepted per-image/layer PCG64 sample, independent of input path."""
    if layer not in LAYERS or not 0 <= take <= spatial:
        raise ValueError('Invalid layer or sample size')
    value = ('balanced-v1|3407|' + input_sha256 + '|' + layer).encode()
    seed = int.from_bytes(hashlib.sha256(value).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed)).choice(spatial, take, replace=False).astype(np.int64)


def align_normal_paths(paths: Iterable[str | Path], profile: Mapping) -> tuple[list[Path], list[str]]:
    """Align native/prepared dataset layouts to the same content-defined order.

    Canonical SHA values are configuration metadata, not redistributed images.
    Duplicate-content files are retained as distinct owners, matching the source
    protocol rather than deduplicating the training set.
    """
    candidates = sorted(Path(p).resolve() for p in paths)
    expected = list(profile['normal_sha256'])
    if len(candidates) != int(profile['normal_count']) or len(expected) != len(candidates):
        raise ValueError(f"Expected {profile['normal_count']} normal images, found {len(candidates)}")
    indexed = defaultdict(deque)
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(path)
        indexed[file_sha256(path)].append(path)
    ordered = []
    for digest in expected:
        if not indexed[digest]:
            raise ValueError('Normal images do not match the selected fixed profile. Check the dataset version and split.')
        ordered.append(indexed[digest].popleft())
    if any(indexed.values()):
        raise ValueError('Unexpected normal images remain after profile alignment')
    return ordered, expected


def normal_split(paths: list[Path], hashes: list[str], profile: Mapping, data_root: str | Path) -> dict:
    """Create the original seeded 70/30 split and alternating calibration A/B."""
    root = Path(data_root).resolve()
    count = len(paths)
    if count != int(profile['normal_count']) or len(hashes) != count:
        raise ValueError('Normal owner count differs from the selected profile')
    order = np.random.default_rng(int(profile['split_seed'])).permutation(count).tolist()
    reference_count = min(max(int(round(count * .70)), 3), count - 3)
    if reference_count != int(profile['reference_count']):
        raise ValueError('Reference count differs from the fixed profile')
    reference, calibration = order[:reference_count], order[reference_count:]
    for key, actual in [('reference_indices', reference), ('calibration_indices', calibration), ('A_indices', calibration[::2]), ('B_indices', calibration[1::2])]:
        if actual != list(profile[key]):
            raise ValueError(f'Fixed split mismatch: {key}')
    inputs = []
    for owner, (path, digest) in enumerate(zip(paths, hashes)):
        absolute = Path(path).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError('Normal image lies outside the supplied dataset root')
        inputs.append({'path': absolute.relative_to(root).as_posix(), 'owner': owner, 'sha256': digest})
    select = lambda indices: [dict(inputs[index]) for index in indices]
    return dict(schema='adnvcnet_normal_split_v1', dataset=profile['dataset'], category=profile['category'],
                normal_inputs=inputs, reference=select(reference), calibration=select(calibration),
                A=select(calibration[::2]), B=select(calibration[1::2]), split_seed=int(profile['split_seed']),
                memory_seed=int(profile['memory_seed']), normal_only=True,
                fitting_subset='A only; B is reserved for development observations')


def build_balanced_bank(teacher_features: Callable, paths: list[Path], hashes: list[str],
                        capacity: int = 24576, per_image_cap: int = 512) -> dict[str, np.ndarray]:
    """Build the separate balanced evaluation bank and its normal-owner metadata.

    This is deliberately separate from the rolling prototype bank used while
    training. Normal calibration excludes prototypes with the current owner.
    """
    import torch
    import torch.nn.functional as F
    if len(paths) != len(hashes) or not paths:
        raise ValueError('Each normal input needs one content hash')
    quotas = balanced_quotas(len(paths), capacity, per_image_cap)
    chunks = {layer: [] for layer in LAYERS}
    owners = {layer: [] for layer in LAYERS}
    locations = {layer: [] for layer in LAYERS}
    with torch.inference_mode():
        for owner, (path, digest, take) in enumerate(zip(paths, hashes, quotas)):
            features = teacher_features(path)
            for layer, grid in LAYERS.items():
                tensor = features[layer].detach()
                if tuple(tensor.shape) != (1, CHANNELS[layer], *grid):
                    raise ValueError(f'{layer}: unexpected teacher feature shape {tuple(tensor.shape)}')
                # Preserve channel-wise float32 normalization before flattening.
                flat = F.normalize(tensor, dim=1)[0].permute(1, 2, 0).reshape(-1, tensor.size(1))
                indices = balanced_positions(digest, layer, grid[0] * grid[1], take)
                selected = flat[torch.from_numpy(indices).to(flat.device)]
                chunks[layer].append(selected.cpu().numpy().copy())
                owners[layer].append(np.full(take, owner, np.int64))
                locations[layer].append(indices)
    result = {}
    total = sum(quotas)
    for layer in LAYERS:
        values = np.concatenate(chunks[layer], axis=0).astype(np.float32, copy=False)
        origin = np.concatenate(owners[layer]); position = np.concatenate(locations[layer])
        if total < capacity:
            values = np.pad(values, ((0, capacity - total), (0, 0)))
            origin = np.pad(origin, (0, capacity - total), constant_values=-1)
            position = np.pad(position, (0, capacity - total), constant_values=-1)
        result.update({f'bank_{layer}': values, f'ptr_{layer}': np.asarray([total % capacity], np.int64),
                       f'full_{layer}': np.asarray([total >= capacity], np.bool_),
                       f'origin_{layer}': origin, f'position_{layer}': position})
    return result


def build_dino_memories(extract_dino: Callable, reference_paths: list[Path], seed: int,
                        patches_per_image: int = 192, max_patches: int = 8192,
                        device: str = 'cuda') -> dict[str, np.ndarray]:
    """Exact global/spatial/CLS construction, supplied with the active features.

    The random generator is shared between per-image sampling and the final
    global subsample. Global normalization occurs on CPU before transfer, while
    spatial/CLS final normalization occurs on the selected execution device,
    matching the preserved original builder.
    """
    import torch
    import torch.nn.functional as F
    if not reference_paths:
        raise ValueError('At least one normal reference is required')
    rng = np.random.default_rng(seed)
    global_chunks, spatial_grids, image_vectors = [], [], []
    expected_grid = None
    with torch.inference_mode():
        for path in reference_paths:
            patches, grid, global_vec, _source = extract_dino(path)
            grid = tuple(grid)
            if expected_grid is None:
                expected_grid = grid
            elif expected_grid != grid:
                raise ValueError(f'DINO grid changed: {expected_grid} versus {grid}')
            spatial_grids.append(patches.reshape(*grid, -1).detach().cpu())
            image_vectors.append(global_vec.detach().cpu())
            count = int(patches.shape[0]);take = min(patches_per_image, count)
            selected = patches
            if take < count:
                indices = rng.choice(count, size=take, replace=False)
                selected = patches.index_select(0, torch.as_tensor(indices, device=patches.device, dtype=torch.long))
            global_chunks.append(selected.detach().cpu())
        global_memory = torch.cat(global_chunks, dim=0)
        if global_memory.shape[0] > max_patches:
            indices = rng.choice(int(global_memory.shape[0]), size=max_patches, replace=False)
            global_memory = global_memory[torch.as_tensor(indices, dtype=torch.long)]
        global_memory = F.normalize(global_memory.float(), dim=-1).to(device)
        spatial_bank = F.normalize(torch.stack(spatial_grids, dim=0).float().to(device), dim=-1)
        image_memory = F.normalize(torch.stack(image_vectors, dim=0).float().to(device), dim=-1)
    return {name: value.detach().cpu().numpy().copy() for name, value in
            [('global_memory', global_memory), ('spatial_bank', spatial_bank), ('image_memory', image_memory)]}


def save_npz(path: str | Path, values: Mapping[str, np.ndarray]) -> dict:
    path = Path(path)
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite resource array: {name}')
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)
    return {'file': path.name, 'sha256': file_sha256(path), 'bytes': path.stat().st_size,
            'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in values.items()}}
