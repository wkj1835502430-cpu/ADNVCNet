# ADNVC_PERFORMANCE_BACKBONE_R1_2026-09-09
# Supports resnet18 / resnet50 / wide_resnet50_2; Raw NPR definition unchanged.
# NVDR_CORE_ADNVCNET_V5_9_LOCKED_FOR_V5_77_ALLCLASS_OVERWRITE
# NVDR_CORE_ADNVCNET_V5_9_LOCKED_FOR_V5_65_OVERWRITE
"""
ADNVCNet / NVDR core model, V5.9 high-recall AUPRO cover.

This file intentionally contains no PatchCore dependency. It implements the
model-side components needed by the official NVDR closed loop:
  1) frozen ImageNet teacher encoder,
  2) trainable student encoder,
  3) scale-wise normal prototype memory,
  4) Raw NPR from cosine discrepancy, optional feature-gradient discrepancy,
     and configurable normal-prototype distance.

Ground-truth masks are never consumed here. They are used only by the eval script
for metrics and qualitative GT display.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm


_BACKBONE_SPECS = {
    "resnet18": {
        "builder": "resnet18",
        "weights": "ResNet18_Weights",
        "channels": {"layer1": 64, "layer2": 128, "layer3": 256},
    },
    "resnet50": {
        "builder": "resnet50",
        "weights": "ResNet50_Weights",
        "channels": {"layer1": 256, "layer2": 512, "layer3": 1024},
    },
    "wide_resnet50_2": {
        "builder": "wide_resnet50_2",
        "weights": "Wide_ResNet50_2_Weights",
        "channels": {"layer1": 256, "layer2": 512, "layer3": 1024},
    },
}
_BACKBONE_ALIASES = {
    "rn18": "resnet18", "r18": "resnet18",
    "rn50": "resnet50", "r50": "resnet50",
    "wr50": "wide_resnet50_2",
    "wrn50": "wide_resnet50_2",
    "wide_resnet50": "wide_resnet50_2",
}


def canonical_backbone(name: str) -> str:
    name = str(name).lower().strip()
    name = _BACKBONE_ALIASES.get(name, name)
    if name not in _BACKBONE_SPECS:
        raise ValueError(
            f"Unsupported backbone={name}. "
            f"Choose from {sorted(_BACKBONE_SPECS)}."
        )
    return name


def _build_resnet(
    backbone: str,
    pretrained: bool,
    allow_random_fallback: bool,
) -> nn.Module:
    spec = _BACKBONE_SPECS[backbone]
    builder = getattr(tvm, spec["builder"])
    if not pretrained:
        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)

    weight_enum = getattr(tvm, spec["weights"], None)
    try:
        if weight_enum is not None:
            return builder(weights=weight_enum.DEFAULT)
        return builder(pretrained=True)
    except Exception as exc:
        if not allow_random_fallback:
            raise RuntimeError(
                f"ImageNet weights for {backbone} could not be loaded. "
                "Do not use a random teacher for the performance experiment. "
                "Make sure torchvision weights are cached/available."
            ) from exc
        print(
            f"[ADNVC WARNING] random {backbone} initialization "
            "because allow_random_fallback=true",
            flush=True,
        )
        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        trainable: bool = True,
        allow_random_fallback: bool = False,
    ):
        super().__init__()
        self.backbone_name = canonical_backbone(backbone)
        m = _build_resnet(
            self.backbone_name,
            pretrained=pretrained,
            allow_random_fallback=allow_random_fallback,
        )
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = (x - self.pixel_mean) / self.pixel_std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        return {"layer1": l1, "layer2": l2, "layer3": l3}


class ScalePrototypeBank(nn.Module):
    """Scale-wise FIFO memory built from normal training features."""

    def __init__(self, layers: Iterable[str], layer_channels: Dict[str, int], bank_size: int = 8192):
        super().__init__()
        self.layers = list(layers)
        self.layer_channels = dict(layer_channels)
        self.bank_size = int(bank_size)
        for layer in self.layers:
            c = int(self.layer_channels[layer])
            self.register_buffer(f"bank_{layer}", torch.zeros(self.bank_size, c))
            self.register_buffer(f"ptr_{layer}", torch.zeros(1, dtype=torch.long))
            self.register_buffer(f"full_{layer}", torch.zeros(1, dtype=torch.bool))

    def _valid_bank(self, layer: str) -> Tuple[torch.Tensor, int]:
        bank = getattr(self, f"bank_{layer}")
        ptr = int(getattr(self, f"ptr_{layer}").item())
        full = bool(getattr(self, f"full_{layer}").item())
        valid = self.bank_size if full else ptr
        return bank[:valid], valid

    @torch.no_grad()
    def reset(self) -> None:
        for layer in self.layers:
            getattr(self, f"bank_{layer}").zero_()
            getattr(self, f"ptr_{layer}").zero_()
            getattr(self, f"full_{layer}").zero_()

    @torch.no_grad()
    def update(self, feats: Dict[str, torch.Tensor], max_samples_per_image: int = 512) -> None:
        for layer in self.layers:
            z = feats[layer].detach()
            flat = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))
            flat = F.normalize(flat, dim=1)
            if flat.size(0) > max_samples_per_image:
                idx = torch.randperm(flat.size(0), device=flat.device)[:max_samples_per_image]
                flat = flat[idx]
            bank = getattr(self, f"bank_{layer}")
            ptr_t = getattr(self, f"ptr_{layer}")
            full_t = getattr(self, f"full_{layer}")
            ptr = int(ptr_t.item())
            n = int(flat.size(0))
            if n <= 0:
                continue
            if n >= self.bank_size:
                bank.copy_(flat[-self.bank_size:])
                ptr_t.fill_(0)
                full_t.fill_(True)
                continue
            end = ptr + n
            if end <= self.bank_size:
                bank[ptr:end].copy_(flat)
            else:
                first = self.bank_size - ptr
                bank[ptr:].copy_(flat[:first])
                bank[: end - self.bank_size].copy_(flat[first:])
                full_t.fill_(True)
            ptr_t.fill_(end % self.bank_size)
            if end >= self.bank_size:
                full_t.fill_(True)

    def distance_map(
        self,
        layer: str,
        feat: torch.Tensor,
        chunk: int = 2048,
        retrieval: str = "nearest",
        top_r: int = 3,
        dispersion_weight: float = 0.0,
    ) -> torch.Tensor:
        """Return a normal-prototype distance map.

        retrieval:
          - ``nearest``: distance to the single nearest prototype;
          - ``topr_mean``: mean distance to the top-r nearest prototypes;
          - ``topr_mean_dispersion``: top-r mean plus a neighborhood
            dispersion penalty.

        All choices are GT-independent and operate only on the prototype bank
        constructed from train_fit normal features.
        """
        bank, valid = self._valid_bank(layer)
        if valid == 0:
            return torch.zeros(
                feat.size(0), 1, feat.size(2), feat.size(3),
                device=feat.device, dtype=feat.dtype,
            )

        mode = str(retrieval).lower().strip()
        aliases = {
            "top_r_mean": "topr_mean",
            "top-r-mean": "topr_mean",
            "top_r_mean_dispersion": "topr_mean_dispersion",
            "top-r-mean-dispersion": "topr_mean_dispersion",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"nearest", "topr_mean", "topr_mean_dispersion"}:
            raise ValueError(f"Unsupported prototype retrieval mode: {retrieval}")

        r = 1 if mode == "nearest" else max(1, min(int(top_r), int(valid)))
        z = F.normalize(feat, dim=1)
        flat = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))
        outs: List[torch.Tensor] = []
        bank_t = bank.t().contiguous()

        for start in range(0, flat.size(0), chunk):
            sim = flat[start:start + chunk] @ bank_t
            top_sim = torch.topk(sim, k=r, dim=1, largest=True, sorted=False).values
            distances = (1.0 - top_sim).clamp_min(0.0)
            mean_distance = distances.mean(dim=1)
            if mode == "topr_mean_dispersion" and r > 1:
                dispersion = distances.std(dim=1, unbiased=False)
                mean_distance = (
                    mean_distance
                    + float(dispersion_weight) * dispersion
                )
            outs.append(mean_distance)

        dist = torch.cat(outs, dim=0).view(
            feat.size(0), feat.size(2), feat.size(3)
        )
        return dist.unsqueeze(1).clamp_min(0.0)


    @torch.no_grad()
    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for layer in self.layers:
            bank, valid = self._valid_bank(layer)
            item: Dict[str, float] = {
                "valid": float(valid),
                "bank_size": float(self.bank_size),
                "ptr": float(int(getattr(self, f"ptr_{layer}").item())),
                "full": float(bool(getattr(self, f"full_{layer}").item())),
            }
            if valid > 0:
                b = bank[:valid]
                item.update({
                    "mean": float(b.mean().detach().cpu()),
                    "std": float(b.std().detach().cpu()),
                    "norm_mean": float(b.norm(dim=1).mean().detach().cpu()),
                    "norm_std": float(b.norm(dim=1).std().detach().cpu()),
                })
            out[layer] = item
        return out


class ADNVCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
        allow_random_fallback: bool = False,
        layers: Optional[List[str]] = None,
        layer_weights: Optional[Dict[str, float]] = None,
        prototype_size: int = 8192,
        max_bank_update_per_image: int = 512,
        prototype_distance_chunk: int = 1024,
        score_disc_weight: float = 1.0,
        score_proto_weight: float = 0.35,
        gradient_mode: str = "none",
        score_grad_weight: float = 0.0,
        gradient_loss_weight: Optional[float] = None,
        prototype_retrieval: str = "nearest",
        prototype_top_r: int = 3,
        prototype_dispersion_weight: float = 0.0,
        temperature: float = 0.07,
        **_: object,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("ADNVCNet expects RGB images with in_channels=3.")
        self.backbone_name = canonical_backbone(backbone)
        self.layer_channels = dict(
            _BACKBONE_SPECS[self.backbone_name]["channels"]
        )
        self.image_size = int(image_size)
        self.layers = list(layers or ["layer1", "layer2", "layer3"])
        for l in self.layers:
            if l not in self.layer_channels:
                raise ValueError(f"Unsupported layer: {l}")
        self.layer_weights = layer_weights or {"layer1": 0.20, "layer2": 0.45, "layer3": 0.35}
        self.max_bank_update_per_image = int(max_bank_update_per_image)
        self.prototype_distance_chunk = max(64, int(prototype_distance_chunk))
        self.score_disc_weight = float(score_disc_weight)
        self.score_proto_weight = float(score_proto_weight)
        self.gradient_mode = str(gradient_mode).lower().strip()
        if self.gradient_mode not in {"none", "spatial_l1", "spatial_l2"}:
            raise ValueError(
                "gradient_mode must be one of: none, spatial_l1, spatial_l2"
            )
        self.score_grad_weight = float(score_grad_weight)
        self.gradient_loss_weight = float(
            self.score_grad_weight
            if gradient_loss_weight is None
            else gradient_loss_weight
        )
        self.prototype_retrieval = str(prototype_retrieval).lower().strip()
        self.prototype_top_r = max(1, int(prototype_top_r))
        self.prototype_dispersion_weight = float(
            prototype_dispersion_weight
        )
        self.temperature = float(temperature)

        self.teacher = ResNetEncoder(
            backbone=self.backbone_name,
            pretrained=pretrained,
            trainable=False,
            allow_random_fallback=allow_random_fallback,
        )
        self.student = ResNetEncoder(
            backbone=self.backbone_name,
            pretrained=pretrained,
            trainable=True,
            allow_random_fallback=allow_random_fallback,
        )
        self.prototype_bank = ScalePrototypeBank(
            self.layers,
            self.layer_channels,
            bank_size=prototype_size,
        )

    @torch.no_grad()
    def reset_memory(self) -> None:
        self.prototype_bank.reset()

    @torch.no_grad()
    def clear_memory(self) -> None:
        self.reset_memory()

    @torch.no_grad()
    def memory_summary(self) -> Dict[str, Dict[str, float]]:
        return self.prototype_bank.summary()

    def update_memory(self, images: torch.Tensor) -> None:
        self.teacher.eval()
        feats_t = self.teacher(images)
        self.prototype_bank.update(feats_t, max_samples_per_image=self.max_bank_update_per_image)

    def _fuse_maps(self, maps: Dict[str, torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        fused = None
        wsum = 0.0
        for layer in self.layers:
            w = float(self.layer_weights.get(layer, 1.0))
            m = maps[layer]
            if m.shape[-2:] != out_size:
                m = F.interpolate(m, size=out_size, mode="bilinear", align_corners=False)
            fused = m * w if fused is None else fused + m * w
            wsum += w
        if fused is None:
            raise RuntimeError("No maps to fuse.")
        return fused / max(wsum, 1e-6)

    @staticmethod
    def _feature_gradient_discrepancy(
        teacher: torch.Tensor,
        student: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Spatial feature-gradient discrepancy on channel-normalized features."""
        if mode == "none":
            return torch.zeros(
                teacher.size(0), 1, teacher.size(2), teacher.size(3),
                device=teacher.device, dtype=teacher.dtype,
            )

        teacher_dx = F.pad(
            teacher[:, :, :, 1:] - teacher[:, :, :, :-1],
            (0, 1, 0, 0),
        )
        student_dx = F.pad(
            student[:, :, :, 1:] - student[:, :, :, :-1],
            (0, 1, 0, 0),
        )
        teacher_dy = F.pad(
            teacher[:, :, 1:, :] - teacher[:, :, :-1, :],
            (0, 0, 0, 1),
        )
        student_dy = F.pad(
            student[:, :, 1:, :] - student[:, :, :-1, :],
            (0, 0, 0, 1),
        )
        delta_x = teacher_dx - student_dx
        delta_y = teacher_dy - student_dy

        if mode == "spatial_l1":
            return 0.5 * (
                delta_x.abs().mean(dim=1, keepdim=True)
                + delta_y.abs().mean(dim=1, keepdim=True)
            )
        return torch.sqrt(
            (
                delta_x.square() + delta_y.square()
            ).mean(dim=1, keepdim=True).clamp_min(1e-12)
        )

    def forward(self, x: torch.Tensor, fg_mask: Optional[torch.Tensor] = None, update_bank: bool = False, compute_proto: Optional[bool] = None, return_debug: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
        if update_bank:
            self.update_memory(x)
        with torch.no_grad():
            feats_t = self.teacher(x)
        feats_s = self.student(x)
        out_size = x.shape[-2:]

        cosine_maps: Dict[str, torch.Tensor] = {}
        gradient_maps: Dict[str, torch.Tensor] = {}
        disc_maps: Dict[str, torch.Tensor] = {}
        proto_maps: Dict[str, torch.Tensor] = {}
        align_losses: List[torch.Tensor] = []

        for layer in self.layers:
            t = feats_t[layer].detach()
            s = feats_s[layer]
            t_n = F.normalize(t, dim=1)
            s_n = F.normalize(s, dim=1)
            cos = (t_n * s_n).sum(dim=1, keepdim=True)
            cosine_disc = (1.0 - cos).clamp_min(0.0)
            gradient_disc = self._feature_gradient_discrepancy(
                t_n, s_n, self.gradient_mode,
            )
            teacher_student_disc = (
                cosine_disc
                + self.score_grad_weight * gradient_disc
            ).clamp_min(0.0)

            cosine_maps[layer] = cosine_disc
            gradient_maps[layer] = gradient_disc
            disc_maps[layer] = teacher_student_disc
            proto_maps[layer] = self.prototype_bank.distance_map(
                layer,
                t,
                chunk=self.prototype_distance_chunk,
                retrieval=self.prototype_retrieval,
                top_r=self.prototype_top_r,
                dispersion_weight=self.prototype_dispersion_weight,
            )
            align_losses.append(
                cosine_disc.mean()
                + self.gradient_loss_weight * gradient_disc.mean()
            )

        cosine_score = self._fuse_maps(cosine_maps, out_size)
        gradient_score = self._fuse_maps(gradient_maps, out_size)
        disc_score = self._fuse_maps(disc_maps, out_size)
        proto_score = self._fuse_maps(proto_maps, out_size)
        raw_score = (
            self.score_disc_weight * disc_score
            + self.score_proto_weight * proto_score
        ).clamp_min(0.0)

        if fg_mask is not None:
            resized_fg = F.interpolate(
                fg_mask.float(), size=out_size, mode="nearest"
            )
            raw_score = raw_score * resized_fg
            cosine_score = cosine_score * resized_fg
            gradient_score = gradient_score * resized_fg
            disc_score = disc_score * resized_fg
            proto_score = proto_score * resized_fg

        result: Dict[str, torch.Tensor] = {
            "raw_score_map": raw_score,
            "score_map": raw_score,
            "cosine_score_map": cosine_score,
            "gradient_score_map": gradient_score,
            "disc_score_map": disc_score,
            "proto_score_map": proto_score,
            "feature_align_loss": torch.stack(align_losses).mean(),
        }
        if return_debug:
            result["cosine_maps"] = cosine_maps  # type: ignore[assignment]
            result["gradient_maps"] = gradient_maps  # type: ignore[assignment]
            result["disc_maps"] = disc_maps  # type: ignore[assignment]
            result["proto_maps"] = proto_maps  # type: ignore[assignment]
        return result


def build_model(cfg: dict, device: Optional[torch.device] = None) -> ADNVCNet:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = ADNVCNet(**model_cfg)
    if device is not None:
        model = model.to(device)
    return model
