"""Portable feature extraction and inference for newly generated resources.

This API accepts explicit checkpoints and normal resources. It does not load
historical run contracts, use test masks, or assert that a new run reproduces
the paper's numbers. Numerical primitives are shared with the local method.
"""
from __future__ import annotations

import copy
import random
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.ad_nvc_net import build_model
from models import dinov2_features as dino_math
from models import normal_tail_mvtec, normal_tail_visa


def dataset_name(profile):
    value = profile.get('dataset_name', profile.get('dataset'))
    if isinstance(value, dict):
        value = value.get('name', value.get('type', value.get('dataset')))
    value = str(value).lower()
    if value not in ('mvtec', 'visa'):
        raise ValueError('Profile must identify dataset mvtec or visa')
    return value


def scoring_operator(dataset):
    if dataset == 'mvtec':
        return normal_tail_mvtec
    if dataset == 'visa':
        return normal_tail_visa
    raise ValueError(dataset)


def _rgb512(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def configure_backend(dataset, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These differ between the current MVTec and VisA numerical profiles.
    torch.backends.cudnn.deterministic = dataset == 'visa'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


class _LocalDino:
    # Reuse the unchanged preprocessing function rather than a second recipe.
    _tensor = dino_math.DinoExtractor._tensor

    def __init__(self, local_repo, weights, device):
        self.device = torch.device(device)
        self.input_size = 518
        self.amp = self.device.type == 'cuda'
        repo = Path(local_repo).resolve()
        weights = Path(weights).resolve()
        if not (repo / 'hubconf.py').is_file():
            raise FileNotFoundError(f'DINOv2 local repository lacks hubconf.py: {repo}')
        if not weights.is_file():
            raise FileNotFoundError(weights)
        torch.manual_seed(13027)
        np.random.seed(13027)
        model = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=False)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.model = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class FeatureExtractor:
    """Extract shared teacher and last-four-layer DINOv2 normal features.

    ``profile`` is the full configuration returned by load_training_profile.
    Local DINOv2 code and weights are explicit, so no implicit download occurs.
    """

    def __init__(self, checkpoint, profile, dino_repo, dino_weights, device='cuda'):
        self.profile = copy.deepcopy(profile)
        self.dataset = dataset_name(profile)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu explicitly for a CPU run')
        configure_backend(self.dataset, int(profile.get('training_seed', profile.get('seed', 3407))))
        config = copy.deepcopy(profile)
        config['model'] = copy.deepcopy(profile['model'])
        config['model']['pretrained'] = False
        evaluation = profile.get('evaluation_model', profile.get('eval_model', {}))
        config['model'].update(copy.deepcopy(evaluation))
        config['model']['pretrained'] = False
        if 'layer_weights' in evaluation:
            config['model']['layer_weights'] = copy.deepcopy(evaluation['layer_weights'])
        config['model'].update(score_disc_weight=1.0, score_proto_weight=0.35,
                               score_grad_weight=0.0, prototype_retrieval='nearest',
                               max_bank_update_per_image=512)
        self.model = build_model(config, device=self.device)
        self.checkpoint_path = Path(checkpoint).resolve()
        checkpoint_data = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' not in checkpoint_data:
            raise ValueError('Checkpoint must contain model_state_dict')
        self.model.load_state_dict(checkpoint_data['model_state_dict'], strict=True)
        self.model.eval()
        del checkpoint_data
        if self.model.layers != ['layer1', 'layer2', 'layer3']:
            raise ValueError('The public profile requires layer1/layer2/layer3')
        self.extractor = _LocalDino(dino_repo, dino_weights, self.device)

    def _cnn_tensor(self, path):
        return torch.from_numpy(_rgb512(path)).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def teacher_features(self, path):
        return self.model.teacher(self._cnn_tensor(path))

    @torch.inference_mode()
    def dino_features(self, path):
        result = dino_math._extract_multilayer_patch_and_global(self.extractor, _rgb512(path))
        if tuple(result[1]) != (37, 37) or result[3] != 'dinov2_last4_normalized_equal_mean_cls':
            raise RuntimeError('Unexpected DINOv2 feature representation')
        return result


def _owner_excluded_distance(bank_module, layer, feat, chunk=2048, retrieval='nearest',
                             top_r=3, dispersion_weight=0.0):
    """The existing nearest-prototype operation with one owner mask added.

    It retains the full GEMM dimensions; filtering prototype columns first
    could change floating-point accumulation choices. The mask is applied
    before the same top-k call used by ScalePrototypeBank.distance_map.
    """
    if retrieval != 'nearest':
        raise ValueError('Normal owner exclusion is defined for nearest retrieval')
    bank, valid = bank_module._valid_bank(layer)
    if valid == 0:
        raise RuntimeError('Cannot calibrate with an empty prototype bank')
    mask = bank_module._public_owner_masks[layer][:valid]
    if bool(mask.all()):
        raise RuntimeError('Owner exclusion leaves no reference prototypes')
    z = F.normalize(feat, dim=1)
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))
    outs = []
    bank_t = bank.t().contiguous()
    for start in range(0, flat.size(0), chunk):
        sim = flat[start:start + chunk] @ bank_t
        sim[:, mask] = float('-inf')
        top_sim = torch.topk(sim, k=1, dim=1, largest=True, sorted=False).values
        distances = (1.0 - top_sim).clamp_min(0.0)
        outs.append(distances.mean(dim=1))
    dist = torch.cat(outs, dim=0).view(feat.size(0), feat.size(2), feat.size(3))
    return dist.unsqueeze(1).clamp_min(0.0)


class PublicInference(FeatureExtractor):
    """Inference using a user's own prototype, memory, and calibration files.

    Calibration is optional during construction so the same raw extraction
    code can be used to fit the six mappings from normal calibration subset A.
    Instances process one image at a time and should not be shared by threads.
    """

    def __init__(self, checkpoint, profile, dino_repo, dino_weights, resources_dir, device='cuda'):
        super().__init__(checkpoint, profile, dino_repo, dino_weights, device)
        self.resources_dir = Path(resources_dir).resolve()
        with np.load(self.resources_dir / 'bank.npz', allow_pickle=False) as data:
            bank_keys = set(self.model.prototype_bank.state_dict())
            self.model.prototype_bank.load_state_dict({k: torch.from_numpy(data[k].copy()) for k in bank_keys}, strict=True)
            self.origins = {layer: np.asarray(data['origin_' + layer], dtype=np.int64).copy() for layer in self.model.layers}
        with np.load(self.resources_dir / 'memories.npz', allow_pickle=False) as data:
            required = {'global_memory', 'spatial_bank', 'image_memory'}
            if set(data.files) != required:
                raise ValueError('memories.npz must contain global_memory, spatial_bank, image_memory')
            self.memory = {k: torch.from_numpy(data[k].copy()).to(self.device) for k in required}
        if not all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in self.memory.values()):
            raise ValueError('DINO memories must be finite float32 arrays')
        if tuple(self.memory['spatial_bank'].shape[1:]) != (37, 37, 384):
            raise ValueError('Spatial memory must have shape [N,37,37,384]')
        self.match_memory = {k: v.to(torch.float64) if self.dataset == 'visa' and k != 'image_memory' else v
                             for k, v in self.memory.items()}
        self.operator = scoring_operator(self.dataset)
        cal_path = self.resources_dir / 'calibration.npz'
        self.calibration = self.operator.load_calibration(cal_path) if cal_path.is_file() else None

    @torch.inference_mode()
    def cnn_raw(self, path, exclude_index=None, return_debug=False):
        bank = self.model.prototype_bank
        original = bank.distance_map
        if exclude_index is not None and int(exclude_index) >= 0:
            owner = int(exclude_index)
            bank._public_owner_masks = {layer: torch.from_numpy(self.origins[layer] == owner).to(self.device) for layer in self.model.layers}
            bank.distance_map = types.MethodType(_owner_excluded_distance, bank)
        try:
            output = self.model(self._cnn_tensor(path), update_bank=False, return_debug=return_debug)
            if return_debug:
                return output
            return output['raw_score_map'][0, 0].cpu().numpy().astype(np.float32)
        finally:
            bank.distance_map = original
            if hasattr(bank, '_public_owner_masks'):
                del bank._public_owner_masks

    @torch.inference_mode()
    def raw_components(self, path, exclude_index=None):
        cnn = self.cnn_raw(path, exclude_index)
        patches, grid, cls, _ = self.dino_features(path)
        query = patches.to(torch.float64) if self.dataset == 'visa' else patches
        values = {
            'cnn': cnn,
            'global': dino_math._raw_global(query, grid, self.match_memory['global_memory'], 512),
            'spatial_r0': dino_math._raw_spatial(query, grid, self.match_memory['spatial_bank'], 0),
            'spatial_r1': dino_math._raw_spatial(query, grid, self.match_memory['spatial_bank'], 1),
            'cls': np.asarray(dino_math.raw_image_distance(cls, self.memory['image_memory'])),
        }
        if not all(np.isfinite(v).all() for v in values.values()):
            raise RuntimeError('Nonfinite raw response')
        return values

    @staticmethod
    def dino_source(raw):
        return {'raw_global': raw['global'], 'raw_spatial_r0': raw['spatial_r0'],
                'raw_spatial_r1': raw['spatial_r1'], 'image_raw_distance': raw['cls']}

    def score_raw(self, raw):
        if self.calibration is None:
            raise RuntimeError('Missing calibration.npz; run tools.calibrate on normal subset A first')
        out = self.operator.predict(raw['cnn'], self.dino_source(raw), self.calibration)
        out['anomaly_map'] = out['pixel_score'].copy()
        return out

    def predict(self, path):
        return self.score_raw(self.raw_components(path))
