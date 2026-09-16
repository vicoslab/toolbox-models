"""Execute the unmodified Toolbox create/export scripts with SDK transport fixtures.

TOOLBOX_HOST selects a checkout. Only SDK network calls are replaced; annotation
RLE, adapter imports, file operations, manifest creation and model training are real.
Python 3.11 gets a test-only relative_to(walk_up=True) compatibility shim because
Toolbox's exporter requires Python >=3.12.
"""
import copy
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
HOST = Path(os.environ.get('TOOLBOX_HOST', ROOT.parents[1] / 'toolbox'))
CLASSES = ['Carbon', 'Film', 'Vacuum']


def vector():
    return dict(id='particle', from_name='radius', to_name='image', type='vector',
                original_width=64, original_height=64, image_rotation=0,
                value={'vertices': [{'x': 25, 'y': 25}, {'x': 37.5, 'y': 25}]})


def ellipse():
    return dict(id='particle', from_name='labels', to_name='image', type='ellipselabels',
                original_width=64, original_height=64, image_rotation=0,
                value=dict(x=25, y=25, radiusX=12.5, radiusY=12.5,
                           rotation=0, ellipselabels=['Particle']))


def brushes():
    from stem_plugin.semantic_results import brush_results
    mask = np.zeros((64, 64), np.uint8)
    mask[:, 21:42] = 1
    mask[:, 42:] = 2
    return brush_results(mask, CLASSES, 'semantic', 'image')


def host_export(tmp_path, monkeypatch, tags, extra_tasks=None):
    import label_studio_sdk
    dataset = tmp_path / 'dataset'
    dataset.mkdir(exist_ok=True)
    for name, value in [('BF.png', 17), ('HAADF.png', 91)]:
        Image.new('L', (64, 64), value).save(dataset / name)
    tasks = [dict(data={'images': [f'https://example/data/local-files/?d=dataset/{n}' for n in ['BF.png', 'HAADF.png']]},
                  annotations=[{'result': tags}], split=split)
             for split in ['Train', 'Validation', 'Test', 'Unassigned']]
    tasks += extra_tasks or []
    payload = json.dumps(tasks).encode()
    exports = SimpleNamespace(create=lambda **kw: SimpleNamespace(status='completed', id=123),
                              download=lambda **kw: [payload])
    monkeypatch.setattr(label_studio_sdk, 'LabelStudio', lambda **kw: SimpleNamespace(projects=SimpleNamespace(exports=exports)))
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
    runpy.run_path(str(HOST / 'apps/ls-utils/export.py'), run_name='__main__')
    return tmp_path / 'export/manifest.json'


def test_host_create_uses_plugin_config(tmp_path, monkeypatch):
    import label_studio_sdk
    import yaml
    calls = {}
    def create(**kw):
        calls.update(kw)
        return SimpleNamespace(id=1)
    client = SimpleNamespace(projects=SimpleNamespace(create=create), ml=SimpleNamespace(create=lambda **kw: None))
    monkeypatch.setattr(label_studio_sdk, 'LabelStudio', lambda **kw: client)
    request = dict(group_size=2, title='STEM', dataset=None, regex_include='.*', regex_exclude='')
    for key, value in dict(LOCAL_FILES_DOCUMENT_ROOT=tmp_path, MODEL_DIR=ROOT,
                           LABEL_STUDIO_USER_TOKEN='fixture-token', CREATION_REQUEST=json.dumps(request)).items():
        monkeypatch.setenv(key, str(value))
    runpy.run_path(str(HOST / 'apps/ls-utils/create.py'), run_name='__main__')
    assert calls['label_config'] == yaml.safe_load((ROOT / 'config.yml').read_text())['config']


@pytest.mark.parametrize('particles,semantic', [(True, False), (False, True), (True, True)])
def test_host_export_loader_training(tmp_path, monkeypatch, particles, semantic):
    import torch
    import mlflow
    from train import Trainer
    from stem_plugin.runtime import StemRuntime
    from stem_plugin.stem_tasks import TaskConfig
    from stem_plugin.toolbox_dataset import ToolboxDataset
    torch.set_num_threads(2)
    tags = ([ellipse()] if particles else []) + (brushes() if semantic else [])
    manifest = host_export(tmp_path, monkeypatch, tags)
    data = json.loads(manifest.read_text())
    assert data['version'] == 4
    assert set(data) == {'version', 'train', 'val', 'test', 'data'}
    assert 'semantic_classes' not in data
    tasks = TaskConfig(particles, semantic)
    ds = ToolboxDataset(manifest, tasks, size=(64, 64))
    sample = ds[0]
    assert sample['image'][:, 0, 0].tolist() == [17., 91., 0.]
    if particles:
        assert ds.items[0]['points'] == [[16., 16., 8.]]
        assert sample['shape_coef'][0, 16, 16] == 8
    else:
        assert 'points' not in ds.items[0]
    if semantic:
        assert ds.items[0]['semantic_classes'] == CLASSES
        assert sample['semantic_segmentation'][0, [0, 21, 42]].tolist() == [0, 1, 2]
    # The held-out test split must not be opened by Trainer.
    data['test'] = [dict(images=['missing-BF', 'missing-HAADF'])]
    manifest.write_text(json.dumps(data))
    options = dict(manifest=str(manifest), nanoparticles=particles, segmentation=semantic,
                   width=64, height=64, batch_size=1, workers=0, epochs=1,
                   save_interval=1, display_interval=1, visualization_samples=1,
                   backbone='resnet18', device='cpu')
    if particles:
        initial = StemRuntime(tasks, backbone='resnet18')
        initial_path = tmp_path / 'initial.pth'
        torch.save(initial.checkpoint(-1), initial_path)
        options['model'] = str(initial_path)
        del initial
    monkeypatch.setenv('MLFLOW_ARTIFACTS_DESTINATION', str(tmp_path / 'artifacts'))
    mlflow.set_tracking_uri('sqlite:///' + str(tmp_path / 'mlflow.db'))
    mlflow.set_experiment('host-export-smoke')
    with mlflow.start_run():
        trainer = Trainer(options)
        trainer.initialize()
        assert trainer.loaders['validation'].dataset.items == data['val']
        trainer.run()
    checkpoint = next((tmp_path / 'artifacts').rglob('checkpoint.pth'))
    restored = StemRuntime(tasks, backbone='resnet18')
    restored.load(torch.load(checkpoint, weights_only=True), inference=True)
    restored.eval()
    result = restored.predict([np.zeros((40, 80, 3), np.uint8)], size=(64, 64))
    assert (result['segmentation'][0] is not None) == semantic
    assert len(list((tmp_path / 'artifacts').rglob('*diagnostics.png'))) == 2


def test_missing_vs_negative(tmp_path, monkeypatch):
    from ls_adapter import export
    assert export([], tmp_path, ['BF', 'HAADF'], False) == {}
    assert export([[]], tmp_path, ['BF', 'HAADF'], False) == {}
    assert 'points' not in export([brushes()], tmp_path, ['BF', 'HAADF'], False)
    reviewed = dict(from_name='reviewed', type='choices', value={'choices': ['Nanoparticles']})
    assert export([[reviewed]], tmp_path, ['BF', 'HAADF'], False) == {'points': []}
    extra = dict(data={'images': ['https://example/data/local-files/?d=dataset/BF.png',
                                  'https://example/data/local-files/?d=dataset/HAADF.png']},
                 annotations=[], split='Train')
    manifest = host_export(tmp_path, monkeypatch, [reviewed], [extra])
    data = json.loads(manifest.read_text())
    assert data['train'][0]['points'] == []
    assert 'points' not in data['train'][1]


def test_overlap_ignore_order_and_duplicate_regions(tmp_path):
    from label_studio_converter.brush import mask2rle
    from ls_adapter import export
    tags = brushes()
    # Carbon overlaps Film, leaving an ignored band, irrespective of region order.
    overlap = copy.deepcopy(tags[0])
    mask = np.zeros((64, 64), np.uint8)
    mask[:, 20:23] = 255
    overlap['value']['rle'] = mask2rle(mask)
    ignored = copy.deepcopy(overlap)
    ignored['value']['brushlabels'] = ['Ignore']
    mask[:] = 0
    mask[0, 0] = 255
    ignored['value']['rle'] = mask2rle(mask)
    tags += [overlap, ignored, copy.deepcopy(tags[0])]
    results = [export([order], tmp_path, ['BF', 'HAADF'], False) for order in [tags, tags[::-1]]]
    assert results[0] == results[1]
    mask = np.array(Image.open(tmp_path / results[0]['semantic_mask']))
    assert mask[0, 0] == 255
    assert mask[1, 20:24].tolist() == [0, 255, 255, 1]
    assert mask[1, 42] == 2


@pytest.mark.parametrize('case', ['multiple', 'pair', 'vertices', 'unknown', 'dimensions', 'rotation', 'rle'])
def test_invalid_exports(tmp_path, case):
    from ls_adapter import export
    tags = brushes() + [vector()]
    annotations, paths = [tags], ['BF', 'HAADF']
    if case == 'multiple': annotations *= 2
    if case == 'pair': paths = ['BF']
    if case == 'vertices': tags[-1]['value']['vertices'].pop()
    if case == 'unknown': tags[0]['value']['brushlabels'] = ['Unknown']
    if case == 'dimensions': tags[0]['original_width'] = 63
    if case == 'rotation': tags[0]['image_rotation'] = 90
    if case == 'rle': tags[0]['original_width'] = 65
    with pytest.raises(ValueError):
        export(annotations, tmp_path, paths, False)


def test_actual_modelargs_flags(monkeypatch):
    import modelargs
    from stem_plugin.task_options import task_config
    for p, s in [('true', 'false'), ('false', 'true'), ('true', 'true')]:
        monkeypatch.setattr(sys, 'argv', ['train', '--nanoparticles', p, '--segmentation', s])
        config = task_config(modelargs.parse(str(ROOT / 'model.json')))
        assert config.nanoparticles == (p == 'true')
        assert config.segmentation == (s == 'true')


def test_exported_dimensions_checked_against_images(tmp_path):
    from ls_adapter import export
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.stem_tasks import TaskConfig
    item = export([[vector()]], tmp_path, ['BF', 'HAADF'], False)
    item['images'] = ['BF.png', 'HAADF.png']
    for name in item['images']:
        Image.new('L', (128, 128)).save(tmp_path / name)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 4, 'train': [item]}))
    with pytest.raises(ValueError, match='annotation dimensions'):
        ToolboxDataset(manifest, TaskConfig(True, False))[0]


@pytest.mark.parametrize('mixed', [False, True])
def test_negative_particles_and_all_ignore_backward(tmp_path, mixed):
    import torch
    from ls_adapter import export
    from stem_plugin.runtime import StemRuntime
    from stem_plugin.stem_tasks import TaskConfig
    from stem_plugin.toolbox_dataset import ToolboxDataset
    torch.set_num_threads(2)
    tag = brushes()[0]
    tag['value']['brushlabels'] = ['Ignore']
    reviewed = dict(from_name='reviewed', value={'choices': ['Nanoparticles']})
    item = export([[tag, reviewed]], tmp_path, ['BF', 'HAADF'], False)
    item['images'] = ['BF.png', 'HAADF.png']
    for name in item['images']:
        Image.new('L', (64, 64), 17).save(tmp_path / name)
    manifest = tmp_path / 'manifest.json'
    items = [item, dict(item, points=[[16, 16, 8]])] if mixed else [item]
    manifest.write_text(json.dumps({'version': 4, 'train': items}))
    tasks = TaskConfig(True, True)
    sample = next(iter(torch.utils.data.DataLoader(ToolboxDataset(manifest, tasks, size=(64, 64)), batch_size=len(items))))
    model = StemRuntime(tasks, backbone='resnet18')
    loss, parts = model.loss(sample)
    assert torch.isfinite(loss)
    assert parts['semantic_loss'].item() == 0
    loss.backward()


def test_class_order_rejected(tmp_path):
    from ls_adapter import export
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.stem_tasks import TaskConfig
    item = export([brushes()], tmp_path, ['BF', 'HAADF'], False)
    item['images'] = ['BF', 'HAADF']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 4, 'train': [item]}))
    with pytest.raises(ValueError, match='class order'):
        ToolboxDataset(manifest, TaskConfig(False, True, ('Vacuum', 'Film', 'Carbon')))
