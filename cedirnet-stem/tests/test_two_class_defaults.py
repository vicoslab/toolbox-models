"""Carbon/Vacuum defaults without rewriting legacy three-class annotations."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
import yaml
from ls_adapter import export
from stem_plugin.stem_tasks import TaskConfig
from stem_plugin.task_options import task_config
from test_multitool_export import region, CONFIG

ROOT = Path(__file__).resolve().parents[1]


def current_config():
    return ET.fromstring(yaml.safe_load((ROOT / 'config.yml').read_text())['config'])


def test_template_has_only_carbon_vacuum_and_ignore():
    labels = current_config().findall(".//Labels[@name='semantic']/Label")
    assert [(t.get('value'), t.get('category')) for t in labels] == [
        ('Carbon', '0'), ('Vacuum', '1'), ('Ignore', '255')]


def test_training_defaults_match_two_class_template():
    options = json.loads((ROOT / 'model.json').read_text())['properties']
    assert json.loads(options['semantic_classes']['default']) == ['Carbon', 'Vacuum']
    assert TaskConfig(False, True).classes == ('Carbon', 'Vacuum')
    assert task_config({'segmentation': True}).classes == ('Carbon', 'Vacuum')


@pytest.mark.parametrize('kind', ['ellipse', 'polygon', 'rectangle', 'brush', 'magicwand'])
@pytest.mark.parametrize('label,category', [('Carbon', 0), ('Vacuum', 1), ('Ignore', 255)])
def test_current_tools_export_two_class_ids(tmp_path, kind, label, category):
    result = export([region(kind, label, True)], tmp_path, ['BF', 'HAADF'], False, current_config())
    assert result['semantic_classes'] == ['Carbon', 'Vacuum']
    assert np.asarray(Image.open(tmp_path / result['semantic_mask']))[30, 20] == category


def test_removed_film_is_rejected_not_silently_remapped(tmp_path):
    with pytest.raises(ValueError, match='configured'):
        export([region('polygon', 'Film', True)], tmp_path, ['BF', 'HAADF'], False, current_config())


def test_explicit_legacy_classes_and_project_config_remain_supported(tmp_path):
    legacy = ['Carbon', 'Film', 'Vacuum']
    assert task_config({'segmentation': True, 'semantic_classes': json.dumps(legacy)}).classes == tuple(legacy)
    assert TaskConfig.from_dict(dict(nanoparticles=False, segmentation=True, classes=legacy)).classes == tuple(legacy)
    result = export([region('polygon', 'Film', True)], tmp_path, ['BF', 'HAADF'], False, CONFIG)
    assert result['semantic_classes'] == legacy
    assert np.asarray(Image.open(tmp_path / result['semantic_mask']))[30, 20] == 1


@pytest.mark.parametrize('particles', [False, True])
def test_two_class_export_runs_real_training_backward(tmp_path, particles):
    import torch
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.runtime import StemRuntime

    torch.set_num_threads(2)
    tags = region('polygon', 'Vacuum', True)
    if particles:
        particle = region('ellipse', 'nanoparticle', True)
        for tag in particle:
            tag['id'] = 'particle'
        tags += particle
    item = export([tags], tmp_path, ['BF.png', 'HAADF.png'], False, current_config())
    item['images'] = ['BF.png', 'HAADF.png']
    for name, value in [('BF.png', 17), ('HAADF.png', 91)]:
        Image.new('L', (100, 80), value).save(tmp_path / name)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(version=4, train=[item])))
    tasks = TaskConfig(particles, True)
    dataset = ToolboxDataset(manifest, tasks, size=(64, 64))
    sample = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    assert sample['semantic_segmentation'].unique().tolist() == [1, 255]
    runtime = StemRuntime(tasks, device='cpu', backbone='resnet18', pretrained=False)
    optimizer = torch.optim.Adam(runtime.parameters(), lr=1e-4)
    before = next(runtime.semantic_model.parameters()).detach().clone()
    runtime.train()
    loss, _ = runtime.loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(runtime.semantic_model.parameters()).detach())
    assert runtime.checkpoint(0)['tasks']['classes'] == ['Carbon', 'Vacuum']
