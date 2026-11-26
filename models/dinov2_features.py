"""Accepted primitive definitions only; original evaluators are not executed."""
from __future__ import annotations
import ast, copy, hashlib, math, os, sys, time, types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
NUM_LAYERS = 4
R21_RADIUS = 1
SIMILARITY_CHUNK = 512
STRICT_RADIUS = 0

class DinoExtractor:

    def __init__(self, model_name: str, input_size: int, device: str, amp: bool, local_repo: str | None) -> None:
        self.model_name = model_name
        self.input_size = int(input_size)
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == 'cuda')
        if self.input_size % 14 != 0:
            raise ValueError('DINOv2 ViT/14 input_size must be divisible by 14.')
        print(f'[R13 ENV] python={sys.version.split()[0]} executable={sys.executable}', flush=True)
        model = _load_dinov2_model_compat(model_name, local_repo)
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def _tensor(self, image: np.ndarray) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32)
        h, w = arr.shape[:2]
        resized = cv2.resize(arr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        x = (x - self.mean) / self.std
        return x.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def patches(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self._tensor(image)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            features = self.model.forward_features(x)
        if isinstance(features, Mapping):
            patch = None
            for key in ('x_norm_patchtokens', 'x_prenorm', 'x_norm'):
                value = features.get(key)
                if torch.is_tensor(value) and value.ndim == 3:
                    patch = value
                    break
            if patch is None:
                raise RuntimeError(f'Could not locate patch tokens in DINO output keys: {list(features.keys())}')
        elif torch.is_tensor(features) and features.ndim == 3:
            patch = features
        else:
            raise RuntimeError(f'Unexpected DINO feature type: {type(features)}')
        patch = patch[0]
        g = self.input_size // 14
        expected = g * g
        if patch.shape[0] < expected:
            n = int(round(math.sqrt(int(patch.shape[0]))))
            if n * n != int(patch.shape[0]):
                raise RuntimeError(f'Cannot reshape DINO tokens: N={patch.shape[0]}, expected {expected}')
            g = n
            expected = g * g
        if patch.shape[0] > expected:
            patch = patch[-expected:]
        patch = F.normalize(patch.float(), dim=-1)
        return (patch, (g, g))

@torch.inference_mode()
def _extract_multilayer_patch_and_global(extractor, image: np.ndarray):
    """
    Fixed last-four-layer DINO representation.

    The official DINOv2 get_intermediate_layers API removes CLS/register tokens
    from its returned patch-token tensor. We nevertheless retain the same grid
    compatibility check used by the existing ADNVC evaluator.
    """
    x = extractor._tensor(image)
    model = extractor.model
    if not hasattr(model, 'get_intermediate_layers'):
        raise RuntimeError('Loaded DINOv2 model has no get_intermediate_layers method. This candidate must not fall back to a different feature recipe.')
    with torch.autocast(device_type=extractor.device.type, dtype=torch.float16, enabled=extractor.amp):
        try:
            outputs = model.get_intermediate_layers(x, n=NUM_LAYERS, reshape=False, return_class_token=True, norm=True)
        except TypeError as exc:
            raise RuntimeError('The local DINOv2 get_intermediate_layers API is incompatible with this fixed candidate. Expected support for n, reshape, return_class_token and norm.') from exc
    outputs = tuple(outputs)
    if len(outputs) != NUM_LAYERS:
        raise RuntimeError(f'Expected exactly {NUM_LAYERS} final DINO blocks, got {len(outputs)}')
    patch_layers: list[torch.Tensor] = []
    cls_layers: list[torch.Tensor] = []
    g = int(extractor.input_size) // 14
    expected = g * g
    resolved_grid: tuple[int, int] | None = None
    for li, item in enumerate(outputs):
        patch, cls = _unwrap_layer_output(item)
        n_tokens = int(patch.shape[0])
        layer_g = g
        layer_expected = expected
        if n_tokens < layer_expected:
            n = int(round(math.sqrt(n_tokens)))
            if n * n != n_tokens:
                raise RuntimeError(f'Layer {li}: cannot reshape {n_tokens} patch tokens; expected at least {layer_expected}.')
            layer_g = n
            layer_expected = n * n
        if n_tokens > layer_expected:
            patch = patch[-layer_expected:]
            n_tokens = int(patch.shape[0])
        if n_tokens != layer_expected:
            raise RuntimeError(f'Layer {li}: patch count {n_tokens} != expected {layer_expected}')
        grid = (layer_g, layer_g)
        if resolved_grid is None:
            resolved_grid = grid
        elif resolved_grid != grid:
            raise RuntimeError(f'DINO grid changed across layers: {resolved_grid} vs {grid}')
        patch_layers.append(F.normalize(patch, dim=-1))
        cls_layers.append(F.normalize(cls, dim=0))
    if resolved_grid is None:
        raise RuntimeError('No DINO intermediate layers were returned.')
    patch_multi = F.normalize(torch.stack(patch_layers, dim=0).mean(dim=0), dim=-1)
    cls_multi = F.normalize(torch.stack(cls_layers, dim=0).mean(dim=0), dim=0)
    source = f'dinov2_last{NUM_LAYERS}_normalized_equal_mean_cls'
    return (patch_multi, resolved_grid, cls_multi, source)

def _unwrap_layer_output(item: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse one official DINOv2 get_intermediate_layers output."""
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        patch, cls = (item[0], item[1])
    else:
        raise RuntimeError('This fixed candidate requires DINOv2 get_intermediate_layers(..., return_class_token=True). The local model did not return (patch_tokens, cls_token) pairs. Do not silently substitute a different representation.')
    if not torch.is_tensor(patch) or not torch.is_tensor(cls):
        raise RuntimeError(f'Unexpected DINOv2 intermediate-layer output types: patch={type(patch)}, cls={type(cls)}')
    if patch.ndim == 3:
        if patch.shape[0] != 1:
            raise RuntimeError(f'Expected DINO batch size 1, got patch shape {tuple(patch.shape)}')
        patch = patch[0]
    elif patch.ndim != 2:
        raise RuntimeError(f'Expected patch tokens [B,N,C] or [N,C], got {tuple(patch.shape)}')
    if cls.ndim == 2:
        if cls.shape[0] != 1:
            raise RuntimeError(f'Expected DINO batch size 1, got CLS shape {tuple(cls.shape)}')
        cls = cls[0]
    elif cls.ndim == 3 and cls.shape[0] == 1 and (cls.shape[1] >= 1):
        cls = cls[0, 0]
    elif cls.ndim != 1:
        raise RuntimeError(f'Expected CLS [B,C] or [C], got {tuple(cls.shape)}')
    return (patch.float(), cls.float())

@torch.inference_mode()
def _raw_global(patches, grid, memory, sim_chunk):
    values = []
    for start in range(0, patches.shape[0], int(sim_chunk)):
        part = patches[start:start + int(sim_chunk)]
        sim = part @ memory.T
        best = sim.max(dim=1).values
        values.append((1.0 - best).float().cpu())
    return torch.cat(values, dim=0).numpy().reshape(grid).astype(np.float32)

@torch.inference_mode()
def _raw_spatial(patches, grid, spatial_bank, radius):
    gh, gw = grid
    qgrid = patches.reshape(gh, gw, -1)
    best = torch.full((gh, gw), -1.0, dtype=torch.float32, device=patches.device)
    r = int(radius)
    for dy in range(-r, r + 1):
        qy0 = max(0, -dy)
        qy1 = min(gh, gh - dy)
        by0 = qy0 + dy
        by1 = qy1 + dy
        for dx in range(-r, r + 1):
            qx0 = max(0, -dx)
            qx1 = min(gw, gw - dx)
            bx0 = qx0 + dx
            bx1 = qx1 + dx
            query = qgrid[qy0:qy1, qx0:qx1, :]
            bank = spatial_bank[:, by0:by1, bx0:bx1, :]
            sim = torch.einsum('hwc,khwc->khw', query, bank)
            local_best = sim.max(dim=0).values
            best[qy0:qy1, qx0:qx1] = torch.maximum(best[qy0:qy1, qx0:qx1], local_best.float())
    if torch.any(best < -0.5):
        raise RuntimeError('Spatial matching left uncovered grid cells.')
    return (1.0 - best).float().cpu().numpy().astype(np.float32)

@torch.inference_mode()
def raw_image_distance(global_vec: torch.Tensor, image_memory: torch.Tensor) -> float:
    vec = F.normalize(global_vec.float(), dim=0)
    sim = image_memory @ vec
    best = float(sim.max().item())
    return float(1.0 - best)
