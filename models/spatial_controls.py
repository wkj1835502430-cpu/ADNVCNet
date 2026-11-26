"""Fixed-total-weight spatial controls and the center-excluded retrieval rule."""
from __future__ import annotations

import numpy as np
import torch

from models.dinov2_features import _raw_spatial
from models.scoring_controls import dino_source, operator

ARMS = ('A0', 'S1_DOUBLE', 'S0_S8')
STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
         (0, 1), (1, -1), (1, 0), (1, 1))
SPATIAL_TOLERANCE = 1e-6
DEFINITIONS = {
    'A0': 'C/2 + (G+S0+S1)/6',
    'S1_DOUBLE': 'C/2 + (G+2*S1)/6',
    'S0_S8': 'C/2 + (G+S0+S8)/6',
}


@torch.inference_mode()
def neighbors(patches, bank):
    """Accepted dot-product arithmetic with only the center offset omitted.

    The candidate bank and query are float64 for VisA and float32 for MVTec.
    As in the original spatial operator, best similarities are accumulated in
    float32. Border queries use only offsets that remain inside the grid.
    """
    h, w = bank.shape[1:3]
    query = patches.reshape(h, w, -1)
    best = torch.full((h, w), -1.0, device=patches.device, dtype=torch.float32)
    for dy, dx in STEPS:
        top, bottom = max(0, -dy), min(h, h - dy)
        left, right = max(0, -dx), min(w, w - dx)
        region = query[top:bottom, left:right]
        candidates = bank[:, top+dy:bottom+dy, left+dx:right+dx]
        local = torch.einsum('hwc,khwc->khw', region, candidates).max(dim=0).values
        best[top:bottom, left:right] = torch.maximum(best[top:bottom, left:right], local.float())
    if not torch.isfinite(best).all() or torch.any(best < -0.5):
        raise ValueError('Neighborhood search failed to cover the grid')
    return (1.0 - best).cpu().numpy()


def score_map(op, variant, raw, dino, cal, s8=None, fit8=None):
    """Keep the original fusion operation order and component mapping order."""
    if variant == 'A0':
        return op.pixel(raw, dino, cal)['pixel_score']
    if variant == 'S1_DOUBLE':
        dino = {**dino, 'raw_spatial_r0': dino['raw_spatial_r1']}
        cal = {**cal, 'spatial_r0': cal['spatial_r1']}
    elif variant == 'S0_S8':
        dino = {**dino, 'raw_spatial_r1': s8}
        cal = {**cal, 'spatial_r1': fit8}
    else:
        raise ValueError(variant)
    return op.pixel(raw, dino, cal)['pixel_score']


def fit_controls(raw_rows, s8_rows, calibration, dataset):
    """Fit S8 first, then each arm's Top mapping using the same normal A."""
    if len(raw_rows) != len(s8_rows) or not raw_rows:
        raise ValueError('Spatial fitting requires matching nonempty normal A rows')
    op = operator(dataset)
    fit8 = op.fit(np.concatenate([op.sample(s8) for s8 in s8_rows]))
    top = {variant: [] for variant in ARMS}
    for raw, s8 in zip(raw_rows, s8_rows, strict=True):
        for variant in ARMS:
            pixel = score_map(op, variant, raw['cnn'], dino_source(raw), calibration, s8, fit8)
            top[variant].append(op.top_fraction(pixel))
    fits = {variant: op.fit(values) for variant, values in top.items()}
    if set(fits['A0']) != set(calibration['top']) or any(
            not np.array_equal(value, calibration['top'][key])
            for key, value in fits['A0'].items()):
        raise ValueError('The same normal A does not reconstruct the original A0 Top mapping')
    return {**fits, 'S8': fit8}, top


def score_control(raw, s8, calibration, fits, dataset, variant):
    if variant not in ARMS:
        raise ValueError(variant)
    op = operator(dataset)
    pixel = score_map(op, variant, raw['cnn'], dino_source(raw), calibration, s8, fits['S8'])
    top = float(op.top_fraction(pixel))
    et = float(op.transform(np.asarray(top), fits[variant]))
    ez = float(op.transform(np.asarray(raw['cls'], np.float64), calibration['cls']))
    return dict(anomaly_map=np.asarray(pixel, np.float64), pixel_score=np.asarray(pixel, np.float64),
                pixel_top1=np.asarray(top, np.float64), image_top_evidence=np.asarray(et, np.float64),
                image_cls_evidence=np.asarray(ez, np.float64), image_score=np.asarray(.5*et+.5*ez, np.float64))


@torch.inference_mode()
def ring_response(runtime, path, expected_s0, expected_s1, anchor=False):
    """Extract S8 and check consistency with the saved S0/S1 responses."""
    patches, grid, _, source = runtime.dino_features(path)
    if tuple(grid) != (37, 37) or tuple(patches.shape) != (1369, 384):
        raise ValueError('Expected the 37x37 DINO ViT-S/14 representation')
    if patches.dtype != torch.float32 or source != 'dinov2_last4_normalized_equal_mean_cls':
        raise ValueError('Unexpected DINO feature precision or representation')
    bank = runtime.match_memory['spatial_bank']
    dtype = torch.float64 if runtime.dataset == 'visa' else torch.float32
    if bank.dtype != dtype:
        raise ValueError('Unexpected dataset-specific spatial matching precision')
    query = patches.to(dtype)
    s8 = neighbors(query, bank)
    error = float(np.max(np.abs(expected_s1 - np.minimum(expected_s0, s8))))
    if anchor:
        for radius, expected in enumerate((expected_s0, expected_s1)):
            actual = _raw_spatial(query, (37, 37), bank, radius)
            error = max(error, float(np.max(np.abs(actual - expected))))
    if not np.isfinite(error) or error > SPATIAL_TOLERANCE:
        raise ValueError(f'Saved spatial response replay differs by {error:.9g}: {path.name}')
    return s8, error
