"""Submitted empty particle annotations are negatives, without new UI controls."""
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
import pytest
import yaml

from ls_adapter import export

ROOT = Path(__file__).resolve().parents[1]


def config():
    return ET.fromstring(yaml.safe_load((ROOT / 'config.yml').read_text())['config'])


def ellipse():
    return dict(from_name='labels', to_name='image', type='ellipselabels',
                original_width=64, original_height=64,
                value=dict(x=25, y=25, radiusX=12.5, radiusY=12.5,
                           ellipselabels=['nanoparticle']))


def brush():
    from label_studio_sdk.converter.brush import mask2rle
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :32] = 255
    return dict(from_name='semantic', to_name='image', type='brushlabels',
                original_width=64, original_height=64,
                value={'brushlabels': ['Carbon'], 'rle': mask2rle(mask)})


def run_export(tmp_path, annotations):
    return export(annotations, tmp_path, ['BF.png', 'HAADF.png'], True, config())


def test_config_has_no_negative_confirmation_control():
    assert config().find(".//Choices[@name='nanoparticle_negative']") is None


def test_submitted_empty_annotation_exports_empty_points(tmp_path):
    assert run_export(tmp_path, [[]]) == {'points': []}


@pytest.mark.parametrize('annotations', [[], None])
def test_no_submission_stays_missing(tmp_path, annotations):
    assert run_export(tmp_path, annotations) == {}


def test_unrelated_choices_are_ignored_without_region_dimensions(tmp_path):
    choice = dict(from_name='split', to_name='image', type='choices', value={'choices': ['Train']})
    assert run_export(tmp_path, [[choice]]) == {'points': []}
    assert run_export(tmp_path, [[choice, ellipse()]]) == run_export(tmp_path, [[ellipse()]])


def test_positive_ellipse_export_is_unchanged(tmp_path):
    assert run_export(tmp_path, [[ellipse()]]) == {
        'points': [[16., 16., 8.]], 'annotation_size': [64, 64]}


def test_semantic_submission_without_particles_is_a_particle_negative(tmp_path):
    exported = run_export(tmp_path, [[brush()]])
    assert exported['points'] == []
    mask = np.array(Image.open(tmp_path / exported['semantic_mask']))
    assert np.all(mask[:, :32] == 0)
    assert np.all(mask[:, 32:] == 255)
    assert exported['annotation_size'] == [64, 64]


@pytest.mark.parametrize('reverse', [False, True])
def test_positive_particle_and_semantic_export_is_unchanged(tmp_path, reverse):
    tags = [ellipse(), brush()]
    exported = run_export(tmp_path, [tags[::-1] if reverse else tags])
    assert exported['points'] == [[16., 16., 8.]]
    assert exported['semantic_classes'] == ['Carbon', 'Vacuum']


def test_exported_negative_is_retained_no_submission_is_filtered(tmp_path):
    from stem_plugin.stem_tasks import TaskConfig
    from stem_plugin.toolbox_dataset import ToolboxDataset
    for name in ('BF.png', 'HAADF.png'):
        Image.new('L', (64, 64), 17).save(tmp_path / name)
    items = [dict(images=['BF.png', 'HAADF.png'], **run_export(tmp_path, [[]])),
             dict(images=['BF.png', 'HAADF.png'], **run_export(tmp_path, []))]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 4, 'train': items}))
    with pytest.warns(UserWarning, match='kept 1 of 2, skipped 1'):
        dataset = ToolboxDataset(manifest, TaskConfig(True, False), size=(64, 64))
    assert dataset.items == items[:1]
    assert not dataset[0]['instance'].any()
    assert not dataset[0]['shape_coef'].any()


def test_actual_host_manifest_distinguishes_submitted_and_missing(tmp_path, monkeypatch):
    """Only the SDK transport is replaced; host export and file operations are real."""
    import os
    import runpy
    import sys
    from types import SimpleNamespace
    import label_studio_sdk
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    for name in ('BF.png', 'HAADF.png'):
        Image.new('L', (64, 64), 17).save(dataset / name)
    images = [f'https://example/data/local-files/?d=dataset/{name}'
              for name in ('BF.png', 'HAADF.png')]
    tasks = [dict(data={'images': images}, split='Train', annotations=annotations)
             for annotations in ([{'result': []}], [{'result': [ellipse()]}], [],
                                 [{'result': [brush()]}])]
    payload = json.dumps(tasks).encode()
    projects = SimpleNamespace(
        get=lambda **kwargs: SimpleNamespace(label_config=ET.tostring(config(), encoding='unicode')),
        exports=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(status='completed', id=123),
            download=lambda **kwargs: [payload]))
    monkeypatch.setattr(label_studio_sdk, 'LabelStudio', lambda **kwargs: SimpleNamespace(projects=projects))
    for key, value in dict(MODEL_FILES=ROOT, LOCAL_FILES_DOCUMENT_ROOT=tmp_path,
                           LABEL_STUDIO_USER_TOKEN='fixture-token', PROJECT_ID='1',
                           LABEL_STUDIO_BASE_DATA_DIR=tmp_path, EXPORT_DIR=tmp_path / 'export').items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv('COMBINE', raising=False)
    if sys.version_info < (3, 12):
        original = Path.relative_to
        def relative_to(self, *other, walk_up=False):
            return Path(os.path.relpath(self, Path(*other))) if walk_up else original(self, *other)
        monkeypatch.setattr(Path, 'relative_to', relative_to)
    host = Path(os.environ.get('TOOLBOX_HOST', ROOT.parents[1] / 'toolbox'))
    runpy.run_path(str(host / 'apps/ls-utils/export.py'), run_name='__main__')
    manifest = json.loads((tmp_path / 'export/manifest.json').read_text())
    assert manifest['version'] == 4
    assert len(manifest['train']) == 4
    assert manifest['train'][0]['points'] == []
    assert manifest['train'][1]['points'] == [[16., 16., 8.]]
    assert 'points' not in manifest['train'][2]
    assert manifest['train'][3]['points'] == []
    assert 'semantic_mask' in manifest['train'][3]
