"""Portable MVTec AD and official VisA 1-class readers.

Training reads normal images only. Test masks are not opened by this module.
All stored paths are relative to the user-supplied data root where possible.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MVTEC_CATEGORIES = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                    'transistor', 'wood', 'zipper')
VISA_CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                   'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
                   'pipe_fryum')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    category: str
    label: int = 0
    mask_path: Path | None = None


def canonical_dataset(dataset: str) -> str:
    value = str(dataset).lower().replace('_ad', '')
    if value not in {'mvtec', 'visa'}:
        raise ValueError('dataset must be mvtec or visa')
    return value


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def order_by_sha(records: Iterable[ImageRecord], sha_list: list[str]) -> list[ImageRecord]:
    """Align normal records to the published canonical content-SHA order.

    Only normal images are read. Repeated identical images are handled by a
    queue, so this also verifies the exact multiplicity of each content hash.
    """
    buckets: dict[str, list[ImageRecord]] = {}
    for record in records:
        buckets.setdefault(file_sha256(record.path), []).append(record)
    out = []
    for value in sha_list:
        if not buckets.get(value):
            raise ValueError('Normal data differ from the published image identity/order: ' + value)
        out.append(buckets[value].pop(0))
    if any(buckets.values()):
        raise ValueError('Additional normal images are not in the published profile')
    return out


def _visa_csv(root: Path, split_file: str | Path | None) -> Path:
    candidates = ([Path(split_file)] if split_file else [
        root / 'split_csv' / '1cls.csv', root / '1cls.csv',
        Path(__file__).with_name('visa_1cls.csv')])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Provide the official VisA split with --split-file /path/to/1cls.csv')


def normal_samples(root: str | Path, dataset: str, category: str,
                   layout: str = 'auto', split_file: str | Path | None = None,
                   canonical_sha256: list[str] | None = None) -> list[ImageRecord]:
    root = Path(root).expanduser().resolve()
    ds = canonical_dataset(dataset)
    valid = MVTEC_CATEGORIES if ds == 'mvtec' else VISA_CATEGORIES
    if category not in valid:
        raise ValueError(f'Unknown {ds} category: {category}')
    prepared = root / category / 'train' / 'good'
    if layout not in {'auto', 'prepared', 'raw'}:
        raise ValueError('layout must be auto, prepared, or raw')
    records = []
    if layout == 'prepared' or (layout == 'auto' and prepared.is_dir()) or ds == 'mvtec':
        if not prepared.is_dir():
            raise FileNotFoundError(prepared)
        paths = sorted(p for p in prepared.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        records = [ImageRecord(p, p.relative_to(root).as_posix(), category) for p in paths]
    else:
        with _visa_csv(root, split_file).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'object', 'split', 'label', 'image', 'mask'}.issubset(reader.fieldnames or []):
                raise ValueError('Invalid official VisA 1-class CSV columns')
            for row in reader:
                if row['object'].strip() != category or row['split'].strip().lower() != 'train':
                    continue
                if row['label'].strip().lower() != 'normal':
                    raise ValueError('The official training split contains a non-normal image')
                rel = Path(row['image'].strip().replace('\\', '/'))
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('CSV image path must remain within data-root')
                path = root / rel
                if not path.is_file():
                    raise FileNotFoundError(path)
                records.append(ImageRecord(path, rel.as_posix(), category))
        # prepare_visa.py preserves stem and lowercases extension. Sorting by
        # that target name matches the original prepared 1-class loader.
        records.sort(key=lambda r: r.path.stem + r.path.suffix.lower())
    if not records:
        raise ValueError('No normal training images found')
    if canonical_sha256 is not None:
        records = order_by_sha(records, canonical_sha256)
    return records


def normal_paths(root: str | Path, dataset: str, category: str, layout: str = 'auto',
                 split_file: str | Path | None = None,
                 canonical_sha256: list[str] | None = None) -> list[Path]:
    return [r.path for r in normal_samples(root, dataset, category, layout, split_file, canonical_sha256)]


def load_rgb_tensor(path: str | Path, image_size: int = 512):
    """Original CNN transform: RGB -> PIL bilinear square -> float32 /255."""
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB').resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class NormalImageDataset:
    def __init__(self, records: list[ImageRecord], image_size: int = 512):
        self.records = records
        self.image_size = int(image_size)
        self.samples = [{'img_path': str(r.path), 'mask_path': None,
                         'category': r.category, 'defect_type': 'good',
                         'is_normal': True} for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return load_rgb_tensor(self.records[index].path, self.image_size)
