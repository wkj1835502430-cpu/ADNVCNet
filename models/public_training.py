"""Normal-only public training primitives extracted from the accepted implementation.

The pseudo-generator and loss primitives retain the original numerical source.
The explicit recipe uses hard copy-paste boundaries, visual strength2, probability
Dice, clean subtraction0.8, and no effective clipping below1e12. No historical
contracts, absolute data paths, pretrained checkpoints, or caches are required.
"""
from __future__ import annotations
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class PseudoBatch:
    images: torch.Tensor
    masks: torch.Tensor
    modes: List[str]
    area_ratios: List[float]

def _odd(value: int, minimum: int = 1) -> int:
    value = max(int(value), int(minimum))
    return value if value % 2 == 1 else value + 1

def _clip_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    return (
        int(np.clip(round(x), 0, width - 1)),
        int(np.clip(round(y), 0, height - 1)),
    )

def _ellipse_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    filled: bool = True,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (
        int(rng.integers(max(1, width // 10), max(2, width * 9 // 10))),
        int(rng.integers(max(1, height // 10), max(2, height * 9 // 10))),
    )
    axes = (
        int(rng.integers(max(2, width // 45), max(3, width // 6))),
        int(rng.integers(max(2, height // 45), max(3, height // 6))),
    )
    thickness = -1 if filled else int(rng.integers(1, max(2, min(height, width) // 80)))
    cv2.ellipse(
        mask,
        center,
        axes,
        float(rng.uniform(0.0, 180.0)),
        0.0,
        360.0,
        1,
        thickness=thickness,
    )
    return mask

def _polyline_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    thin: bool,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    count = int(rng.integers(2, 7))
    x = float(rng.integers(0, width))
    y = float(rng.integers(0, height))
    points = [_clip_point(x, y, width, height)]
    for _ in range(count - 1):
        x += float(rng.normal(0.0, width * 0.12))
        y += float(rng.normal(0.0, height * 0.12))
        points.append(_clip_point(x, y, width, height))
    thickness = (
        int(rng.integers(1, 4))
        if thin
        else int(rng.integers(4, max(5, min(height, width) // 28)))
    )
    cv2.polylines(
        mask,
        [np.asarray(points, dtype=np.int32)],
        isClosed=False,
        color=1,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return (mask > 0).astype(np.uint8)

def _arc_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (
        int(rng.integers(width // 8, max(width // 8 + 1, width * 7 // 8))),
        int(rng.integers(height // 8, max(height // 8 + 1, height * 7 // 8))),
    )
    axes = (
        int(rng.integers(max(4, width // 18), max(5, width // 4))),
        int(rng.integers(max(4, height // 18), max(5, height // 4))),
    )
    start = float(rng.uniform(0.0, 300.0))
    end = min(360.0, start + float(rng.uniform(25.0, 120.0)))
    cv2.ellipse(
        mask,
        center,
        axes,
        float(rng.uniform(0.0, 180.0)),
        start,
        end,
        1,
        thickness=int(rng.integers(1, 4)),
        lineType=cv2.LINE_AA,
    )
    return (mask > 0).astype(np.uint8)

def _micro_dot_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    dots = int(rng.integers(2, 18))
    radius_max = max(2, min(height, width) // 70)
    for _ in range(dots):
        center = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        radius = int(rng.integers(1, radius_max + 1))
        cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask

def _blob_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        mask = np.maximum(mask, _ellipse_mask(height, width, rng, filled=True))
    kernel = np.ones((_odd(int(rng.integers(3, 12))),) * 2, dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.uint8)

def _fit_area(
    mask: np.ndarray,
    min_ratio: float,
    max_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = mask.shape
    total = float(height * width)
    min_area = max(1, int(total * float(min_ratio)))
    max_area = max(min_area, int(total * float(max_ratio)))
    current = int(mask.sum())

    if current <= 0:
        side = max(2, int(np.sqrt((min_area + max_area) * 0.5)))
        top = max(0, (height - side) // 2)
        left = max(0, (width - side) // 2)
        mask[top:top + min(side, height), left:left + min(side, width)] = 1
        current = int(mask.sum())

    if current < min_area:
        for kernel_size in (3, 5, 7, 9, 13, 17, 23, 31):
            mask = cv2.dilate(
                mask,
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
                iterations=1,
            )
            current = int(mask.sum())
            if current >= min_area:
                break

    if current > max_area:
        ys, xs = np.flatnonzero(mask).reshape(-1, 1), None
        flat = np.flatnonzero(mask.reshape(-1))
        keep_count = max_area
        keep = rng.choice(flat, size=keep_count, replace=False)
        reduced = np.zeros(height * width, dtype=np.uint8)
        reduced[keep] = 1
        mask = reduced.reshape(height, width)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

    return (mask > 0).astype(np.uint8)

def _make_mask(
    mode: str,
    height: int,
    width: int,
    min_ratio: float,
    max_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode in {"scratch"}:
        mask = _polyline_mask(height, width, rng, thin=False)
    elif mode in {"thin_scratch", "line_gap"}:
        mask = _polyline_mask(height, width, rng, thin=True)
    elif mode == "thin_arc":
        mask = _arc_mask(height, width, rng)
    elif mode == "micro_dot":
        mask = _micro_dot_mask(height, width, rng)
    elif mode == "edge_ring":
        mask = _ellipse_mask(height, width, rng, filled=False)
    else:
        mask = _blob_mask(height, width, rng)
    return _fit_area(mask, min_ratio, max_ratio, rng)

def _blur_image(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        image.astype(np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    ).astype(np.float32)

def _replacement(
    mode: str,
    source: np.ndarray,
    donor: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = source.shape[:2]
    local_mean = source.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)

    if mode == "copy_paste":
        shift_y = int(rng.integers(-height // 3, height // 3 + 1))
        shift_x = int(rng.integers(-width // 3, width // 3 + 1))
        return np.roll(donor, shift=(shift_y, shift_x), axis=(0, 1))

    if mode in {"erase", "line_gap"}:
        blurred = _blur_image(source, float(rng.uniform(4.0, 11.0)))
        strength = float(rng.uniform(0.65, 1.0))
        return strength * local_mean + (1.0 - strength) * blurred

    if mode == "local_blur":
        return _blur_image(source, float(rng.uniform(2.0, 8.0)))

    if mode == "low_contrast":
        blurred = _blur_image(source, float(rng.uniform(1.5, 5.0)))
        strength = float(rng.uniform(0.55, 0.88))
        return strength * local_mean + (1.0 - strength) * blurred

    if mode == "color_shift":
        shift = rng.uniform(-0.32, 0.32, size=(1, 1, 3)).astype(np.float32)
        scale = rng.uniform(0.65, 1.35, size=(1, 1, 3)).astype(np.float32)
        return np.clip(source * scale + shift, 0.0, 1.0)

    if mode == "texture_noise":
        noise = rng.normal(0.0, 1.0, size=source.shape).astype(np.float32)
        noise = _blur_image(noise, float(rng.uniform(0.6, 2.4)))
        noise /= max(float(noise.std()), 1e-6)
        amplitude = float(rng.uniform(0.05, 0.24))
        return np.clip(source + amplitude * noise, 0.0, 1.0)

    noise = rng.normal(
        0.0,
        float(rng.uniform(0.035, 0.18)),
        size=source.shape,
    ).astype(np.float32)
    shift = rng.uniform(-0.22, 0.22, size=(1, 1, 3)).astype(np.float32)
    return np.clip(source + noise + shift, 0.0, 1.0)

class _BasePseudoAnomalyGenerator:
    """Generate pseudo images and synthetic masks from normal images only."""

    def __init__(
        self,
        pseudo_cfg: Dict[str, object],
        category: str,
        seed: int = 3407,
    ) -> None:
        self.cfg = dict(pseudo_cfg or {})
        self.category = str(category)
        self.seed = int(seed)
        category_modes = self.cfg.get("category_modes", {}) or {}
        selected = category_modes.get(self.category)
        self.modes = list(selected or self.cfg.get("modes", ["blob"]))
        if not self.modes:
            self.modes = ["blob"]

    def generate(
        self,
        images: torch.Tensor,
        epoch: int,
        batch_index: int,
    ) -> PseudoBatch:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("Expected images with shape [B,3,H,W].")
        if images.device.type != "cpu":
            raise ValueError("Generate pseudo anomalies before moving images to CUDA.")

        batch, _channels, height, width = images.shape
        seed = (
            self.seed
            + int(epoch) * 1_000_003
            + int(batch_index) * 10_007
        )
        rng = np.random.default_rng(seed)

        output = images.detach().clone().float()
        masks = torch.zeros((batch, 1, height, width), dtype=torch.float32)
        modes_used: List[str] = []
        area_ratios: List[float] = []

        enable_prob = float(self.cfg.get("enable_prob", 0.98))
        min_ratio = float(self.cfg.get("min_area_ratio", 0.002))
        max_ratio = float(self.cfg.get("max_area_ratio", 0.085))
        enabled_count = 0

        images_np = images.permute(0, 2, 3, 1).numpy().astype(np.float32)

        for index in range(batch):
            enabled = bool(rng.random() <= enable_prob)
            if not enabled:
                modes_used.append("none")
                area_ratios.append(0.0)
                continue

            mode = str(rng.choice(self.modes))
            mask = _make_mask(
                mode,
                height,
                width,
                min_ratio,
                max_ratio,
                rng,
            )
            donor_index = int(rng.integers(0, batch))
            if batch > 1 and donor_index == index:
                donor_index = (index + 1) % batch

            source = images_np[index]
            donor = images_np[donor_index]
            replacement = _replacement(mode, source, donor, mask, rng)

            alpha = mask.astype(np.float32)
            if mode in {
                "blob", "copy_paste", "texture_noise",
                "color_shift", "local_blur", "low_contrast",
            }:
                # Preserve blur-kernel RNG consumption.
                blurred_alpha = cv2.GaussianBlur(
                    alpha,
                    (_odd(int(rng.integers(3, 10))),) * 2,
                    sigmaX=0.0,
                )
                blurred_alpha = np.clip(blurred_alpha, 0.0, 1.0)
                if mode == "copy_paste":
                    alpha = mask.astype(np.float32)
                else:
                    alpha = blurred_alpha

            alpha = alpha[..., None]
            mixed = np.clip(
                source * (1.0 - alpha) + replacement * alpha,
                0.0,
                1.0,
            )
            output[index] = torch.from_numpy(
                mixed.transpose(2, 0, 1).astype(np.float32)
            )
            masks[index, 0] = torch.from_numpy(mask.astype(np.float32))
            modes_used.append(mode)
            area_ratios.append(float(mask.mean()))
            enabled_count += 1

        # Avoid an all-negative pseudo batch, which makes the margin term invalid.
        if enabled_count == 0 and batch > 0:
            mode = str(self.modes[0])
            rng_force = np.random.default_rng(seed + 97)
            mask = _make_mask(
                mode,
                height,
                width,
                min_ratio,
                max_ratio,
                rng_force,
            )
            source = images_np[0]
            donor = images_np[1 % batch]
            replacement = _replacement(mode, source, donor, mask, rng_force)
            alpha = mask.astype(np.float32)[..., None]
            mixed = np.clip(source * (1.0 - alpha) + replacement * alpha, 0.0, 1.0)
            output[0] = torch.from_numpy(mixed.transpose(2, 0, 1))
            masks[0, 0] = torch.from_numpy(mask.astype(np.float32))
            modes_used[0] = mode
            area_ratios[0] = float(mask.mean())

        return PseudoBatch(
            images=output,
            masks=masks,
            modes=modes_used,
            area_ratios=area_ratios,
        )

def dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probability_flat = probability.contiguous().flatten(1)
    target_flat = target.contiguous().flatten(1)
    intersection = (probability_flat * target_flat).sum(dim=1)
    denominator = probability_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()

def positive_negative_scores(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    positive_scores: List[torch.Tensor] = []
    negative_scores: List[torch.Tensor] = []
    for score_map, mask_map in zip(probability, target):
        positive = score_map[mask_map > 0.5]
        negative = score_map[mask_map <= 0.5]
        if positive.numel() > 0:
            positive_scores.append(positive.mean())
        if negative.numel() > 0:
            negative_scores.append(negative.mean())
    zero = probability.sum() * 0.0
    return (
        torch.stack(positive_scores).mean()
        if positive_scores else zero,
        torch.stack(negative_scores).mean()
        if negative_scores else zero,
    )

def probability_from_score(
    score: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return (
        1.0 - torch.exp(-float(temperature) * score.clamp_min(0.0))
    ).clamp(1e-6, 1.0 - 1e-6)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PseudoAnomalyGenerator(_BasePseudoAnomalyGenerator):
    """Accepted pseudo-generator with explicit visual-strength amplification."""
    def generate(self, images, epoch, batch_index):
        batch = super().generate(images, epoch, batch_index)
        clean = images.detach().float()
        pseudo = batch.images.detach().float()
        batch.images = torch.clamp(clean + 2.0 * (pseudo - clean), 0.0, 1.0)
        return batch


def load_training_profile(dataset: str, category: str, seed: int = 3407) -> dict:
    """Return a portable copy of the exact category recipe, without data paths."""
    from datasets.public_data import canonical_dataset
    ds = canonical_dataset(dataset)
    path = Path(__file__).resolve().parents[1] / 'configs/training_profiles.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    if int(seed) not in document['seeds']:
        raise ValueError('Published training seeds are3407,3408,3409')
    try:
        cfg = copy.deepcopy(document['datasets'][ds][category])
    except KeyError as exc:
        raise ValueError(f'Unknown dataset/category: {ds}/{category}') from exc
    cfg['protocol'] = copy.deepcopy(document['protocol'])
    cfg['project'] = {'name': 'ADNVCNet', 'seed': int(seed)}
    cfg['train']['seed'] = int(seed)
    cfg['training_seed'] = int(seed)
    return cfg


def initialize_training_model(cfg: dict, device='cpu', imagenet_weights=None):
    """Use the same fixed ImageNet weights for teacher and student.

    By default torchvision downloads/caches its official ResNet18 weights.
    A local official torchvision ImageNet state dictionary may be supplied for
    offline use; it is not an ADNVCNet trained checkpoint.
    """
    from models.ad_nvc_net import build_model
    import hashlib
    expected_imagenet_sha = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    def verify_imagenet(path):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != expected_imagenet_sha:
            raise ValueError('Initialization must be the official ResNet18 IMAGENET1K_V1 weights; SHA256 differs')
    cached_weights = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    if imagenet_weights is not None:
        verify_imagenet(imagenet_weights)
    elif cached_weights.is_file():
        verify_imagenet(cached_weights)
    model_cfg = copy.deepcopy(cfg)
    if imagenet_weights is not None:
        model_cfg['model']['pretrained'] = False
    model = build_model(model_cfg, device=torch.device(device))
    if imagenet_weights is not None:
        state = torch.load(Path(imagenet_weights), map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        for encoder in [model.teacher, model.student]:
            selected = {key: state[key] for key in encoder.state_dict()}
            encoder.load_state_dict(selected, strict=True)
    model.teacher.eval()
    for parameter in model.teacher.parameters():
        parameter.requires_grad_(False)
    assert all(torch.equal(a, b) for a, b in zip(model.teacher.state_dict().values(), model.student.state_dict().values()))
    return model


def loss_terms(clean_out: dict, pseudo_out: dict, pseudo_masks: torch.Tensor,
               pseudo_cfg: dict) -> dict:
    """The effective accepted probability-Dice loss, without any test data."""
    supervised_score = (1.75 * (pseudo_out['disc_score_map']
                                - 0.8 * clean_out['disc_score_map'].detach())).clamp_min(0.0)
    probability = probability_from_score(supervised_score, float(pseudo_cfg['positive_temperature']))
    normal_align = 0.5 * (clean_out['feature_align_loss'] + pseudo_out['feature_align_loss'])
    normal_score = 0.5 * (clean_out['raw_score_map'].clamp_min(0.0).mean()
                          + pseudo_out['raw_score_map'].clamp_min(0.0).mean())
    target = pseudo_masks  # Effective target_dilate_kernel=1, no mask expansion.
    pseudo_bce = F.binary_cross_entropy(probability, target)
    pseudo_dice = dice_loss(probability, target, smooth=1.0)
    pos_score, neg_score = positive_negative_scores(probability, target)
    margin_loss = F.relu(torch.as_tensor(float(pseudo_cfg['margin']),
                                       dtype=probability.dtype, device=probability.device)
                         - pos_score + neg_score)
    total = (float(pseudo_cfg['normal_align_weight']) * normal_align
             + float(pseudo_cfg['normal_score_weight']) * normal_score
             + float(pseudo_cfg['pseudo_bce_weight']) * pseudo_bce
             + float(pseudo_cfg['pseudo_dice_weight']) * pseudo_dice
             + float(pseudo_cfg['pseudo_score_weight']) * margin_loss)
    return {'train_loss': total, 'normal_align': normal_align, 'normal_score': normal_score,
            'pseudo_bce': pseudo_bce, 'pseudo_dice': pseudo_dice, 'margin_loss': margin_loss,
            'pos_score': pos_score, 'neg_score': neg_score}


@torch.no_grad()
def rebuild_training_bank(model, dataset, device='cpu') -> dict:
    """Original training FIFO-bank pass over all sorted normal images once."""
    from torch.utils.data import DataLoader
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    model.eval()
    model.reset_memory()
    for images in loader:
        model.update_memory(images.to(device, non_blocking=True))
    return model.memory_summary()


def optimizer_for(model, cfg: dict):
    return torch.optim.AdamW([p for p in model.student.parameters() if p.requires_grad],
                             lr=float(cfg['train']['lr']),
                             weight_decay=float(cfg['train']['weight_decay']))


def training_step(model, clean_cpu, generator, cfg, optimizer, device,
                  epoch: int, batch_index: int):
    pseudo = generator.generate(clean_cpu, epoch=epoch, batch_index=batch_index)
    clean = clean_cpu.to(device, non_blocking=True)
    synthetic = pseudo.images.to(device, non_blocking=True)
    masks = pseudo.masks.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    clean_out = model(clean, update_bank=False, compute_proto=True)
    pseudo_out = model(synthetic, update_bank=False, compute_proto=True)
    losses = loss_terms(clean_out, pseudo_out, masks, cfg['pseudo'])
    if not torch.isfinite(losses['train_loss']):
        raise FloatingPointError('Non-finite training loss')
    losses['train_loss'].backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.student.parameters() if p.requires_grad], max_norm=1e12)
    optimizer.step()
    values = {k: float(v.detach().cpu()) for k, v in losses.items()}
    values['grad_norm'] = float(norm.detach().cpu())
    return values, pseudo.modes
