"""Installer regression and opt-in real environment compatibility gates."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_installer_resolves_host_and_model_in_one_transaction():
    setup = (ROOT / 'setup.sh').read_text()
    assert '--override "$dir/dependency-overrides.txt"' in setup
    assert '-r "$dir/requirements.txt"' in setup
    assert '--no-deps' not in setup
    assert 'opencv-python-headless' not in setup
    overrides = (ROOT / 'dependency-overrides.txt').read_text()
    assert 'label-studio-sdk==2.0.0' in overrides
    assert 'timm==0.6.13' in overrides


def test_existing_install_is_never_overwritten(tmp_path):
    target = tmp_path / 'cedirnet-stem'
    target.mkdir()
    artifact = target / 'user-model.pt'
    artifact.write_bytes(b'preserve user weights')
    result = subprocess.run(['bash', str(ROOT / 'setup.sh')], env={**os.environ, 'TOOLBOX_CACHE': str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Refusing to overwrite' in result.stderr
    assert artifact.read_bytes() == b'preserve user weights'


@pytest.mark.skipif(os.environ.get('CEDIRNET_STEM_VERIFY_DEPENDENCIES') != '1', reason='requires actual setup environment')
def test_real_dependency_stack():
    from importlib.metadata import distributions, version
    import cv2
    import numpy as np
    import torch
    import timm
    import segmentation_models_pytorch
    import modelargs
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse
    from label_studio_sdk import LabelStudio
    from label_studio_sdk.converter.brush import mask2rle, decode_rle
    from label_studio_converter.brush import decode_rle as legacy_decode
    assert version('label-studio-sdk') == '2.0.0'
    assert version('numpy') == '1.26.4'
    assert timm.__version__ == '0.6.13'
    assert torch.__version__.startswith('2.7.0')
    opencv = {d.metadata['Name'] for d in distributions() if d.metadata['Name'].startswith('opencv-')}
    assert opencv == {'opencv-python'}
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    encoded = mask2rle(mask)
    assert np.array_equal(decode_rle(encoded), legacy_decode(encoded))
    assert cv2.resize(mask, (4, 4)).shape == (4, 4)

    # Check the complete installed graph, allowing only the documented SMP pin
    # replacement. A working import must not hide another unresolved conflict.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    installed = {canonicalize_name(d.metadata['Name']): d for d in distributions()}
    conflicts = []
    for name, dist in installed.items():
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            target = canonicalize_name(req.name)
            if name == 'segmentation-models-pytorch' and target == 'timm':
                assert str(req.specifier) == '==0.6.12'
                continue
            if target not in installed or installed[target].version not in req.specifier:
                conflicts.append(f'{name}: {raw}')
    assert not conflicts, conflicts
