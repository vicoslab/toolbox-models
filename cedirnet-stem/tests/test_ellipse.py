"""Ellipse annotation UI and export contracts, independent of model weights."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from ls_adapter import export

ROOT = Path(__file__).resolve().parents[1]


def ellipse():
    return dict(id='particle', from_name='labels', to_name='image', type='ellipselabels',
                original_width=200, original_height=100, image_rotation=0,
                value=dict(x=25, y=50, radiusX=10, radiusY=10, rotation=35,
                           ellipselabels=['Particle']))


def test_ellipse_center_and_mean_pixel_radius(tmp_path):
    # x/y are CENTER percentages, not the top-left; rotation does not move it.
    result = export([[ellipse()]], tmp_path, ['BF', 'HAADF'], False)
    assert result == {'annotation_size': [200, 100], 'points': [[50, 50, 15]]}
    assert export([[ellipse(), ellipse()]], tmp_path, ['BF', 'HAADF'], False) == result


@pytest.mark.parametrize('field,value', [('x', -1), ('x', 100), ('radiusX', 0),
    ('radiusY', -1), ('radiusX', float('nan')), ('rotation', float('inf'))])
def test_invalid_ellipse(tmp_path, field, value):
    tag = ellipse()
    tag['value'][field] = value
    with pytest.raises(ValueError):
        export([[tag]], tmp_path, ['BF', 'HAADF'], False)


def test_toggles_never_invent_negative_supervision(tmp_path):
    tasks = dict(from_name='annotation_tasks', type='choices',
                 value={'choices': ['Nanoparticles', 'Segmentation']})
    assert export([[tasks]], tmp_path, ['BF', 'HAADF'], False) == {}
    assert export([[]], tmp_path, ['BF', 'HAADF'], False) == {}
    negative = dict(from_name='particle_review', type='choices', value={'choices': ['No nanoparticles']})
    assert export([[tasks, negative]], tmp_path, ['BF', 'HAADF'], False) == {'points': []}
    with pytest.raises(ValueError, match='negative'):
        export([[negative, ellipse()]], tmp_path, ['BF', 'HAADF'], False)


def test_ellipse_preannotation_export_roundtrip(tmp_path):
    from stem_plugin.serving import preannotation
    response = dict(tasks={'nanoparticles': True}, centers=[[[.25, .5]]],
                    radii=[[10]], scores=[[.8]], segmentation=[None])
    config = {'labels': dict(type='EllipseLabels', labels=['Particle'], to_name=['image'])}
    annotation = preannotation(response, 0, (200, 100), config)
    tag, = annotation['result']
    assert tag['type'] == 'ellipselabels'
    assert tag['value'] == dict(x=25, y=50, radiusX=5, radiusY=10, rotation=0, ellipselabels=['Particle'])
    assert export([annotation['result']], tmp_path, ['BF', 'HAADF'], False)['points'] == [[50, 50, 10]]


def test_non_circular_ellipse_export_dataset(tmp_path):
    import json
    from PIL import Image
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.stem_tasks import TaskConfig
    item = export([[ellipse()]], tmp_path, ['BF', 'HAADF'], False)
    item['images'] = ['BF.png', 'HAADF.png']
    for name, value in zip(item['images'], [17, 91]):
        Image.new('L', (200, 100), value).save(tmp_path / name)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 4, 'train': [item]}))
    sample = ToolboxDataset(manifest, TaskConfig(True, False), size=(200, 100))[0]
    assert sample['image'][:, 0, 0].tolist() == [17, 91, 0]
    assert sample['shape_coef'][0, 50, 50] == 15


def test_sidebar_ui_and_hotkeys():
    root = ET.fromstring(yaml.safe_load((ROOT / 'config.yml').read_text())['config'])
    assert root.attrib == {'className': 'stem-annotation'}
    assert root.find('.//Choices') is None
    assert root.find('.//Choice') is None
    row = root.find('View')
    assert row.attrib['style'] == 'display: flex; flex-direction: row; align-items: flex-start; gap: 16px;'
    left, right = row.findall('View')
    assert left.attrib['style'] == 'flex: 1 1 0%; min-width: 0;'
    assert right.attrib['style'] == 'display: flex; flex-direction: row; align-items: flex-start; gap: 8px; flex: 0 0 380px;'
    assert left.find('Image').attrib == dict(name='image', valueList='$images', zoom='true', gallery='true', shared='true')
    assert root.find('Image') is None
    assert [p.attrib['value'] for p in right.findall('View/Collapse/Panel')] == ['Particle instances', 'Segmentation']
    assert all(c.attrib == dict(open='true', bordered='true') for c in root.findall('.//Collapse'))
    assert root.find('.//EllipseLabels/Label').attrib == dict(value='PtCo', alias='nanoparticle', background='#14c850', hotkey='p')
    assert [x.attrib['hotkey'] for x in root.findall('.//BrushLabels/Label')] == ['1', '2', '3', '4']
    assert root.find(".//Label[@value='Ignore']").attrib['hint'] == 'Use when unsure which category.'
    assert root.find(".//Panel[@value='Segmentation']/Magicwand").attrib == dict(name='wand', toName='image')
    assert root.find('.//Vector') is None


@pytest.mark.parametrize('label', ['nanoparticle', 'Particle'])
def test_configured_and_legacy_labels(tmp_path, label):
    tag = ellipse()
    tag['value']['ellipselabels'] = [label]
    assert export([[tag]], tmp_path, ['BF', 'HAADF'], False)['points'] == [[50, 50, 15]]


@pytest.mark.parametrize('labels', [['PtCo'], ['Unknown'], [], ['nanoparticle', 'Particle']])
def test_unknown_particle_labels_rejected(tmp_path, labels):
    tag = ellipse()
    tag['value']['ellipselabels'] = labels
    with pytest.raises(ValueError, match='configured particle'):
        export([[tag]], tmp_path, ['BF', 'HAADF'], False)


def test_real_sdk_alias_preannotation_roundtrip(tmp_path):
    from label_studio_sdk._extensions.label_studio_tools.core.label_config import parse_config
    from stem_plugin.serving import preannotation
    config = parse_config(yaml.safe_load((ROOT / 'config.yml').read_text())['config'])
    assert config['labels']['labels'] == ['nanoparticle']
    assert config['labels']['inputs'][0]['valueList'] == 'images'
    response = dict(tasks={'nanoparticles': True}, centers=[[[.25, .5]]],
                    radii=[[10]], scores=[[.8]], segmentation=[None])
    annotation = preannotation(response, 0, (200, 100), config)
    assert annotation['result'][0]['value']['ellipselabels'] == ['nanoparticle']
    assert export([annotation['result']], tmp_path, ['BF', 'HAADF'], False)['points'] == [[50, 50, 10]]
