"""Ordinary inherited permissions for temporary test directories on Windows."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import uuid


@contextmanager
def test_directory():
    base = Path(os.environ.get('ADNVCNET_TEST_TEMP_ROOT', tempfile.gettempdir())).resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / ('adnvcnet_inference_test_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != base or not root.name.startswith('adnvcnet_inference_test_'):
            raise RuntimeError('Refusing cleanup outside the temporary test directory')
        shutil.rmtree(root)
