"""SDK-backed predictions for shared geometry and separate class controls."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
from label_studio_sdk._extensions.label_studio_tools.core.label_config import parse_config
from label_studio_sdk.converter.brush import decode_rle

from stem_plugin.semantic_results import encode_mask
from stem_plugin.serving import preannotation


ROOT = Path(__file__).resolve().parents[1]
MASK = np.array([[0, 1, 2, 255], [2, 1, 0, 255]], dtype=np.uint8)
CLASSES = ['Carbon', 'Film', 'Vacuum']


def config_xml(*, combined=False, aliases=False):
    particle = 'EllipseLabels' if combined else 'Labels'
    semantic = 'BrushLabels' if combined else 'Labels'
    labels = ''.join(
        f'<Label value="{name}" category="{category}"'
        + (f' alias="{alias}"' if aliases else '') + '/>'
        for name, category, alias in [('Carbon', 0, 'c'), ('Film', 1, 'f'),
                                      ('Vacuum', 2, 'v'), ('Ignore', 255, 'i')])
    tools = '' if combined else '''<Ellipse name="points" toName="image"/>
        <Polygon name="polygon" toName="image"/>
        <Brush name="brush" toName="image"/>
        <Magicwand name="wand" toName="image"/>'''
    return f'''<View><View><Image name="image" valueList="$images" gallery="true" shared="true"/></View>
        <{particle} name="labels" toName="image"><Label value="PtCo" alias="nanoparticle"/></{particle}>
        <{semantic} name="semantic" toName="image">{labels}</{semantic}>{tools}</View>'''


def response(*, particles=True, semantic=True):
    return dict(tasks={'nanoparticles': particles, 'segmentation': semantic},
                centers=[[[.25, .5], [.75, .5]]], radii=[[1, .5]], scores=[[.8, .6]],
                segmentation=[encode_mask(MASK, CLASSES) if semantic else None])


def alpha(result):
    return np.asarray(decode_rle(result['value']['rle']), dtype=np.uint8).reshape(2, 4, 4)[:, :, 3]


def assert_pair(geometry, label, kind, tool, label_control, values):
    assert geometry['type'] == kind
    assert geometry['from_name'] == tool
    assert label['type'] == 'labels'
    assert label['from_name'] == label_control
    assert label['value'] == {**geometry['value'], 'labels': values}
    assert not any(key.endswith('labels') for key in geometry['value'])
    for field in ('id', 'to_name', 'original_width', 'original_height', 'image_rotation', 'item_index'):
        assert geometry[field] == label[field]
    assert geometry['to_name'] == 'image'
    assert (geometry['original_width'], geometry['original_height']) == (4, 2)
    assert geometry['item_index'] == 0


def test_shared_particle_geometry_uses_serialized_alias():
    config = parse_config(config_xml())
    assert config['labels']['inputs'][0]['valueList'] == 'images'
    assert config['labels']['labels'] == ['nanoparticle']
    prediction = preannotation(response(semantic=False), 0, (4, 2), config)
    regions = prediction['result']
    assert len(regions) == 4
    for geometry, label in zip(regions[::2], regions[1::2]):
        assert_pair(geometry, label, 'ellipse', 'points', 'labels', ['nanoparticle'])
    assert regions[0]['value'] == dict(x=25, y=50, radiusX=25, radiusY=50, rotation=0)
    assert len({region['id'] for region in regions}) == 2
    assert prediction['score'] == pytest.approx(.7)


@pytest.mark.parametrize('aliases', [False, True])
def test_shared_semantic_geometry_is_editor_loadable(aliases):
    config = parse_config(config_xml(aliases=aliases))
    regions = preannotation(response(particles=False), 0, (4, 2), config)['result']
    assert len(regions) == 6
    serialized = ['c', 'f', 'v'] if aliases else CLASSES
    for index, (geometry, label) in enumerate(zip(regions[::2], regions[1::2])):
        assert_pair(geometry, label, 'brush', 'brush', 'semantic', [serialized[index]])
        assert geometry['value']['format'] == 'rle'
        np.testing.assert_array_equal(alpha(geometry), (MASK == index).astype(np.uint8)*255)
    assert len({region['id'] for region in regions}) == 3


def test_shared_joint_results_have_distinct_region_ids():
    regions = preannotation(response(), 0, (4, 2), parse_config(config_xml()))['result']
    assert len(regions) == 10
    labels = [region for region in regions if region['type'] == 'labels']
    assert len({region['id'] for region in labels}) == 5
    assert {region['from_name'] for region in labels} == {'labels', 'semantic'}


@pytest.mark.parametrize('aliases', [False, True])
def test_legacy_combined_controls_preserve_geometry_and_aliases(aliases):
    regions = preannotation(response(), 0, (4, 2), parse_config(config_xml(combined=True, aliases=aliases)))['result']
    assert [region['type'] for region in regions] == ['ellipselabels']*2 + ['brushlabels']*3
    assert regions[0]['value']['ellipselabels'] == ['nanoparticle']
    serialized = ['c', 'f', 'v'] if aliases else CLASSES
    for index, region in enumerate(regions[2:]):
        assert region['value']['brushlabels'] == [serialized[index]]
        np.testing.assert_array_equal(alpha(region), (MASK == index).astype(np.uint8)*255)
    assert all(region['item_index'] == 0 for region in regions)


@pytest.mark.parametrize('tool, particles, semantic', [('Ellipse', True, False), ('Brush', False, True)])
@pytest.mark.parametrize('ambiguous', [False, True])
def test_shared_predictions_reject_missing_or_ambiguous_geometry(tool, particles, semantic, ambiguous):
    xml = ET.fromstring(config_xml())
    element = xml.find(tool)
    if ambiguous:
        ET.SubElement(xml, tool, name='duplicate', toName='image')
    else:
        xml.remove(element)
    with pytest.raises(ValueError, match=tool):
        preannotation(response(particles=particles, semantic=semantic), 0, (4, 2), parse_config(ET.tostring(xml, encoding='unicode')))


def test_batch_index_is_not_gallery_frame_index():
    data = response()
    for field in ('centers', 'radii', 'scores', 'segmentation'):
        data[field].insert(0, [] if field != 'segmentation' else None)
    regions = preannotation(data, 1, (4, 2), parse_config(config_xml()))['result']
    assert len(regions) == 10
    assert all(region['item_index'] == 0 for region in regions)


def test_semantic_empty_mask_emits_no_regions():
    data = response(particles=False)
    data['segmentation'] = [encode_mask(np.full_like(MASK, 255), CLASSES)]
    assert preannotation(data, 0, (4, 2), parse_config(config_xml()))['result'] == []


def test_particle_tool_resolves_label_target_not_unrelated_image():
    xml = ET.fromstring(config_xml())
    ET.SubElement(xml, 'Image', name='other', value='$other')
    ET.SubElement(xml, 'Ellipse', name='unrelated', toName='other')
    regions = preannotation(response(semantic=False), 0, (4, 2), parse_config(ET.tostring(xml, encoding='unicode')))['result']
    assert len(regions) == 4
    assert regions[0]['from_name'] == 'points'


@pytest.mark.parametrize('combined', [False, True])
@pytest.mark.parametrize('aliases', [False, True])
def test_sdk_preannotation_export_roundtrip(tmp_path, combined, aliases):
    from ls_adapter import export

    xml = config_xml(combined=combined, aliases=aliases)
    prediction = preannotation(response(), 0, (4, 2), parse_config(xml))
    # Exercise serialized predictions, not just in-memory dictionaries.
    tags = json.loads(json.dumps(prediction))['result']
    exported = export([tags], tmp_path, ['BF.png', 'HAADF.png'], False, ET.fromstring(xml))
    assert exported['points'] == [[1, 1, 1], [3, 1, .5]]
    assert exported['annotation_size'] == [4, 2]
    assert exported['semantic_classes'] == CLASSES
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / exported['semantic_mask'])), MASK)


@pytest.mark.parametrize('particles, semantic', [(True, False), (False, True), (True, True)])
def test_repository_sdk_config_preannotation_export_roundtrip(tmp_path, particles, semantic):
    import yaml
    from ls_adapter import export

    xml = yaml.safe_load((ROOT / 'config.yml').read_text())['config']
    config = parse_config(xml)
    assert config['labels']['type'] == config['semantic']['type'] == 'Labels'
    assert config['labels']['labels'] == ['nanoparticle']
    assert config['labels']['inputs'][0]['valueList'] == 'images'
    assert {name: config[name]['type'] for name in ['points', 'polygon', 'brush', 'wand']} == {
        'points': 'Ellipse', 'polygon': 'Polygon', 'brush': 'Brush', 'wand': 'Magicwand'}
    current_classes = ['Carbon', 'Vacuum']
    current_mask = np.array([[0, 1, 1, 255], [1, 1, 0, 255]], dtype=np.uint8)
    data = response(particles=particles, semantic=semantic)
    data['segmentation'] = [encode_mask(current_mask, current_classes) if semantic else None]
    prediction = preannotation(data, 0, (4, 2), config)
    tags = json.loads(json.dumps(prediction))['result']
    assert all(tag['from_name'] in config for tag in tags)
    assert all(tag['type'] == config[tag['from_name']]['type'].lower() for tag in tags)
    assert all(tag['item_index'] == 0 for tag in tags)
    exported = export([tags], tmp_path, ['BF.png', 'HAADF.png'], False, ET.fromstring(xml))
    assert exported['points'] == ([[1, 1, 1], [3, 1, .5]] if particles else [])
    if semantic:
        assert exported['semantic_classes'] == current_classes
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / exported['semantic_mask'])), current_mask)
    else:
        assert 'semantic_mask' not in exported
