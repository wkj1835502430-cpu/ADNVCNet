"""Small path-independent helpers for public source workflows."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def relative_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path leaves the selected root: {relative}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_profile(resources_dir):
    profile = read_json(Path(resources_dir) / 'profile.json')
    return profile.get('configuration', profile.get('profile', profile))
