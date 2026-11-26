from __future__ import annotations
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

def _pixel_auroc(score_maps: list[np.ndarray], gts: list[np.ndarray]) -> float:
    y = np.concatenate([(gt > 0.5).reshape(-1) for gt in gts]).astype(np.uint8)
    s = np.concatenate([m.reshape(-1) for m in score_maps]).astype(np.float64)
    return float(roc_auc_score(y, s))

def _image_auroc(scores: list[float], labels: list[int]) -> float:
    return float(roc_auc_score(np.asarray(labels, dtype=np.uint8), np.asarray(scores, dtype=np.float64)))

def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask_u8 = (np.asarray(mask) > 0.5).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    return [labels == i for i in range(1, count)]

def _aupro(score_maps: list[np.ndarray], gts: list[np.ndarray], max_fpr: float=0.3, thresholds: int=200) -> float:
    scores = np.concatenate([m.reshape(-1) for m in score_maps]).astype(np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float('nan')
    normal_pixels = int(sum((np.count_nonzero(np.asarray(gt) <= 0.5) for gt in gts)))
    if normal_pixels <= 0:
        return float('nan')
    components_per_image = [_connected_components(gt) for gt in gts]
    total_components = sum((len(x) for x in components_per_image))
    if total_components == 0:
        return float('nan')
    hi = float(np.max(finite))
    lo = float(np.min(finite))
    if hi <= lo + 1e-12:
        return 0.0
    ts = np.linspace(hi + 1e-07, lo - 1e-07, int(thresholds))
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for t in ts:
        fp = 0
        overlaps: list[float] = []
        for score, gt, components in zip(score_maps, gts, components_per_image):
            pred = score >= t
            normal = gt <= 0.5
            fp += int(np.count_nonzero(pred & normal))
            for comp in components:
                denom = int(np.count_nonzero(comp))
                if denom > 0:
                    overlaps.append(float(np.count_nonzero(pred & comp)) / float(denom))
        fpr = fp / float(normal_pixels)
        pro = float(np.mean(overlaps)) if overlaps else 0.0
        if fpr <= max_fpr + 1e-12:
            points.append((fpr, pro))
    points.sort(key=lambda p: p[0])
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        if xs and abs(x - xs[-1]) < 1e-12:
            ys[-1] = max(ys[-1], y)
        else:
            xs.append(x)
            ys.append(y)
    if xs[-1] < max_fpr:
        xs.append(max_fpr)
        ys.append(ys[-1])
    xs_np = np.asarray(xs, dtype=np.float64)
    ys_np = np.asarray(ys, dtype=np.float64)
    keep = xs_np <= max_fpr + 1e-12
    area = float(np.trapz(ys_np[keep], xs_np[keep]))
    return area / float(max_fpr)
