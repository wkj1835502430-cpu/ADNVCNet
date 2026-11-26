"""Fixed-model score controls using the original pixel and Top arithmetic.

Only normal calibration subset A fits a changed Top mapping. No model is
trained here, and no test label/mask enters a scoring configuration.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from models import normal_tail_mvtec, normal_tail_visa

DEFAULT_VARIANTS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'O1')
SPATIAL_VARIANTS = ('S_NO_S0', 'S_NO_S1', 'S_NO_BOTH')
MAPPING_VARIANTS = ('RAW_PIXEL', 'LINEAR_Q_PIXEL', 'TAIL_PIXEL')
NO_FIT_VARIANTS = ('T1', 'DINO_G')
INDIVIDUAL_MAPPING_VARIANTS = ('ID_CNN', 'ID_GLOBAL', 'ID_S0', 'ID_S1', 'ID_TOP', 'ID_CLS')
RETAINED_TOP_VARIANTS = ('A5', 'A6', 'ID_TOP', 'ID_CLS')
ALL_VARIANTS = DEFAULT_VARIANTS + SPATIAL_VARIANTS + MAPPING_VARIANTS + NO_FIT_VARIANTS + INDIVIDUAL_MAPPING_VARIANTS
RAW_COMPONENTS = ('cnn', 'global', 'spatial_r0', 'spatial_r1')

DESCRIPTIONS = {
 'A0': 'Full method: C/2 + (G+S0+S1)/6; fuse then Top1%; combine Top and CLS equally.',
 'A1': 'Remove CNN pixel response: (G+S0+S1)/3; refit Top on the same normal A.',
 'A2': 'Remove all DINO pixel responses: C; retain CLS and refit Top.',
 'A3': 'Remove global G: C/2 + S0/4 + S1/4; refit Top.',
 'A4': 'Remove both spatial responses and redistribute weights: C/2 + G/2; refit Top.',
 'A5': 'Remove CLS image term: retain the A0 pixel map and use its Top evidence.',
 'A6': 'Remove Top image term: retain the A0 pixel map and use CLS evidence.',
 'O1': 'Retain A0 pixel map; aggregate each calibrated plane before weighted combination; refit this Top statistic.',
 'S_NO_S0': 'Set S0 to zero while keeping C/2 + (G+S0+S1)/6 coefficients fixed; refit Top.',
 'S_NO_S1': 'Set S1 to zero with all other coefficients fixed; refit Top.',
 'S_NO_BOTH': 'Set S0 and S1 to zero with all other coefficients fixed; refit Top.',
 'RAW_PIXEL': 'Identity mappings for four pixel responses; unchanged CLS calibration; refit Top.',
 'LINEAR_Q_PIXEL': 'Map normal A sampled 90th/99.5th percentiles to 0/0.1, extend linearly without clipping; unchanged CLS; refit Top.',
 'TAIL_PIXEL': 'Empirical upper-tail pixel maps, unchanged CLS; refit Top on the same A (equivalent to A0).',
 'T1': 'All six mappings are identity: raw pixel fusion, raw Top and raw CLS; no mapping is fitted.',
 'DINO_G': 'Shared frozen DINO global raw distance, linearly resized to 512; raw Top1% image score; no CNN/CLS/mapping fit.',
 'ID_CNN': 'Identity CNN pixel mapping only; retain other mappings and refit Top on the same normal A.',
 'ID_GLOBAL': 'Identity global DINO pixel mapping only; retain other mappings and refit Top on the same normal A.',
 'ID_S0': 'Identity same-position pixel mapping only; retain other mappings and refit Top on the same normal A.',
 'ID_S1': 'Identity neighborhood pixel mapping only; retain other mappings and refit Top on the same normal A.',
 'ID_TOP': 'Identity image Top mapping only; retain the A0 pixel map and calibrated CLS term.',
 'ID_CLS': 'Identity image CLS mapping only; retain the A0 pixel map and calibrated Top term.'}


def operator(dataset):
    if dataset == 'mvtec':return normal_tail_mvtec
    if dataset == 'visa':return normal_tail_visa
    raise ValueError(dataset)


def dino_source(raw):
    return dict(raw_global=raw['global'], raw_spatial_r0=raw['spatial_r0'],
                raw_spatial_r1=raw['spatial_r1'], image_raw_distance=raw['cls'])


def baseline(raw, calibration, dataset):
    return operator(dataset).predict(raw['cnn'], dino_source(raw), calibration)


def pixel_variant(tail, variant):
    """Original A0–A6 pixel_variant arithmetic, with additional fixed controls."""
    if variant in ('A0', 'A5', 'A6', 'O1', 'TAIL_PIXEL', 'ID_TOP', 'ID_CLS'):
        return np.asarray(tail['pixel_score'], dtype=np.float64)
    c = tail['e_cnn']
    g, s0, s1 = [cv2.resize(tail[key], (512, 512), interpolation=cv2.INTER_LINEAR)
                 for key in ('e_global_patch', 'e_spatial_r0_patch', 'e_spatial_r1_patch')]
    if variant == 'A1':return (g + s0 + s1) / 3.0
    if variant == 'A2':return c.copy()
    if variant == 'A3':return .5*c + .25*s0 + .25*s1
    if variant == 'A4':return .5*c + .5*g
    if variant in SPATIAL_VARIANTS:
        if variant in ('S_NO_S0', 'S_NO_BOTH'):s0 = np.zeros_like(s0)
        if variant in ('S_NO_S1', 'S_NO_BOTH'):s1 = np.zeros_like(s1)
        return .5*c + .5*(g + s0 + s1)/3.
    raise ValueError(variant)


def separate_top(tail, dataset):
    """Original O1: Top of each upsampled calibrated plane, then same weights."""
    op = operator(dataset)
    large = [np.asarray(tail['e_cnn'], np.float64)]
    large.extend(cv2.resize(np.asarray(tail[k], np.float64), (512, 512), interpolation=cv2.INTER_LINEAR)
                 for k in ('e_global_patch', 'e_spatial_r0_patch', 'e_spatial_r1_patch'))
    tops = np.asarray([op.top_fraction(a) for a in large], np.float64)
    return float(.5*tops[0] + .5*(tops[1] + tops[2] + tops[3])/3.)


def quantile_map(values, quantiles):
    lo, hi = map(float, quantiles)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f'Degenerate normal quantiles: {lo}, {hi}')
    return 0.1*(values-lo)/(hi-lo)


def fuse_raw_planes(planes):
    g, s0, s1 = [cv2.resize(a, (512, 512), interpolation=cv2.INTER_LINEAR) for a in planes[1:]]
    return .5*planes[0] + .5*(g+s0+s1)/3.


def control_pixel(raw, tail, variant, quantiles=None):
    if variant in ('RAW_PIXEL', 'T1'):
        return fuse_raw_planes([np.asarray(raw[name], np.float64) for name in RAW_COMPONENTS])
    if variant == 'DINO_G':
        return cv2.resize(np.asarray(raw['global'], np.float64), (512, 512), interpolation=cv2.INTER_LINEAR)
    if variant == 'LINEAR_Q_PIXEL':
        if quantiles is None:raise ValueError('Missing normal quantile fits')
        return fuse_raw_planes([quantile_map(np.asarray(raw[name], np.float64), quantiles[name]) for name in RAW_COMPONENTS])
    if variant in INDIVIDUAL_MAPPING_VARIANTS[:4]:
        cnn = np.asarray(raw['cnn'], np.float64) if variant == 'ID_CNN' else tail['e_cnn']
        patches = [tail[key] for key in ('e_global_patch', 'e_spatial_r0_patch', 'e_spatial_r1_patch')]
        for index, (name, control) in enumerate(zip(RAW_COMPONENTS[1:], ('ID_GLOBAL', 'ID_S0', 'ID_S1'))):
            if variant == control:patches[index] = np.asarray(raw[name], np.float64)
        up = [cv2.resize(patch, (512, 512), interpolation=cv2.INTER_LINEAR) for patch in patches]
        return .5*cnn + .5*(up[0]+up[1]+up[2])/3.
    return pixel_variant(tail, variant)


def fit_controls(raw_factory, calibration, dataset, variants=DEFAULT_VARIANTS):
    """raw_factory() must replay the same ordered normal A inputs each time."""
    variants = tuple(variants)
    if not variants or any(v not in ALL_VARIANTS for v in variants):
        raise ValueError('Unknown or empty control selection')
    if all(v in NO_FIT_VARIANTS for v in variants):
        return dict(schema='adnvcnet_fixed_score_controls_v1', variants=variants, quantiles=None,
                    top_fits={}, normal_A_count=0, original_top_reconstructed_exact=False)
    op = operator(dataset)
    quantiles = None
    if 'LINEAR_Q_PIXEL' in variants:
        pools = {name: [] for name in RAW_COMPONENTS}
        for raw in raw_factory():
            for name in RAW_COMPONENTS:pools[name].append(op.sample(np.asarray(raw[name], np.float64)))
        quantiles = {}
        for name, parts in pools.items():
            if not parts:raise ValueError('No normal calibration A inputs')
            values = np.concatenate(parts)
            quantiles[name] = np.percentile(values, [90., 99.5], method='linear')
            quantile_map(quantiles[name], quantiles[name])
    top_values = {v: [] for v in variants if v not in RETAINED_TOP_VARIANTS + NO_FIT_VARIANTS}
    original_top_values = []
    count = 0
    for raw in raw_factory():
        tail = baseline(raw, calibration, dataset)
        original_top_values.append(float(tail['pixel_top1']))
        for variant in top_values:
            value = separate_top(tail, dataset) if variant == 'O1' else op.top_fraction(control_pixel(raw, tail, variant, quantiles))
            top_values[variant].append(float(value))
        count += 1
    if not count:raise ValueError('No normal calibration A inputs')
    rebuilt = op.fit(original_top_values)
    if any(not np.array_equal(rebuilt[key], calibration['top'][key]) for key in rebuilt):
        raise ValueError('Stored normal A responses do not reconstruct the original Top calibration')
    top_fits = {variant: op.fit(values) for variant, values in top_values.items()}
    for variant in RETAINED_TOP_VARIANTS:
        if variant in variants:top_fits[variant] = calibration['top']
    return dict(schema='adnvcnet_fixed_score_controls_v1', variants=variants, quantiles=quantiles,
                top_fits=top_fits, normal_A_count=count, original_top_reconstructed_exact=True)


def score_control(raw, calibration, dataset, variant, fits, tail=None):
    if variant not in fits['variants']:raise ValueError('Variant was not selected in this control run')
    op = operator(dataset)
    if variant in NO_FIT_VARIANTS:
        pixel = control_pixel(raw, None, variant)
        top = float(op.top_fraction(pixel))
        cls = float(raw['cls']) if variant == 'T1' else 0.
        image = .5*top + .5*cls if variant == 'T1' else top
        return dict(anomaly_map=np.asarray(pixel, np.float64), pixel_score=np.asarray(pixel, np.float64),
                    pixel_top1=np.asarray(top, np.float64), image_top_evidence=np.asarray(top, np.float64),
                    image_cls_evidence=np.asarray(cls, np.float64), image_score=np.asarray(image, np.float64))
    tail = baseline(raw, calibration, dataset) if tail is None else tail
    pixel = control_pixel(raw, tail, variant, fits.get('quantiles'))
    top = separate_top(tail, dataset) if variant == 'O1' else float(op.top_fraction(pixel))
    if variant in ('A0', 'A5', 'A6', 'ID_CLS'):
        evidence = float(tail['image_top_evidence'])
    elif variant == 'ID_TOP':
        evidence = top
    else:
        evidence = float(op.transform(np.asarray(top), fits['top_fits'][variant]))
    cls = float(raw['cls']) if variant == 'ID_CLS' else float(tail['image_cls_evidence'])
    if variant == 'A0':image = float(tail['image_score'])
    elif variant == 'A5':image = evidence
    elif variant == 'A6':image = cls
    else:image = .5*evidence + .5*cls
    return dict(anomaly_map=np.asarray(pixel, np.float64), pixel_score=np.asarray(pixel, np.float64),
                pixel_top1=np.asarray(top, np.float64), image_top_evidence=np.asarray(evidence, np.float64),
                image_cls_evidence=np.asarray(cls, np.float64), image_score=np.asarray(image, np.float64))


def fit_arrays(fits):
    result = {f'top__{v}__{k}': np.asarray(a) for v, part in fits['top_fits'].items() for k, a in part.items()}
    if fits.get('quantiles') is not None:
        result.update({f'q__{name}': np.asarray(value) for name, value in fits['quantiles'].items()})
    return result


def load_fit_arrays(values, variants, normal_A_count):
    top_fits = {v: {key.split('__', 2)[2]: values[key].copy() for key in values if key.startswith('top__' + v + '__')} for v in variants}
    quantiles = {name: values['q__' + name].copy() for name in RAW_COMPONENTS} if 'q__cnn' in values else None
    return dict(variants=tuple(variants), top_fits=top_fits, quantiles=quantiles, normal_A_count=normal_A_count)
