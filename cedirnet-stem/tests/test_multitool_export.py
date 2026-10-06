"""Label-driven export contract for shared Label Studio drawing tools."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
import yaml
from label_studio_sdk.converter.brush import mask2rle
from ls_adapter import export

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ET.fromstring('''<View><Image name="image" valueList="$images"/>
<Labels name="labels" toName="image"><Label value="PtCo" alias="nanoparticle"/></Labels>
<Labels name="semantic" toName="image">
<Label value="Carbon" category="0"/><Label value="Film" category="1"/>
<Label value="Vacuum" category="2"/><Label value="Ignore" category="255"/>
</Labels></View>''')


def region(kind, label, separate=False):
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:40, 10:30] = 255
    geometry = {
        'ellipse': dict(x=20, y=37.5, radiusX=10, radiusY=12.5, rotation=0),
        'rectangle': dict(x=10, y=25, width=20, height=25, rotation=0),
        'polygon': dict(points=[[10, 25], [30, 25], [30, 50], [10, 50]]),
        'brush': dict(format='rle', rle=mask2rle(mask)),
        'magicwand': dict(format='rle', rle=mask2rle(mask)),
    }[kind]
    control = 'labels' if label in ('nanoparticle', 'Particle') else 'semantic'
    tag = dict(id='region', from_name='tool' if separate else control,
               to_name='image', item_index=0, original_width=100,
               original_height=80, image_rotation=0, type=kind, value=geometry)
    if separate:
        partner = dict(id='region', from_name=control, to_name='image', item_index=0,
                       type='labels', value={'labels': [label]})
        return [partner, tag]
    tag['type'] = 'brushlabels' if kind == 'magicwand' else kind + 'labels'
    tag['value'][tag['type']] = [label]
    return [tag]


def run(tmp_path, tags, config=CONFIG):
    return export([tags], tmp_path, ['BF.png', 'HAADF.png'], False, config)


@pytest.mark.parametrize('kind', ['ellipse', 'rectangle', 'polygon', 'brush', 'magicwand'])
@pytest.mark.parametrize('label,category', [('Carbon', 0), ('Film', 1), ('Vacuum', 2), ('Ignore', 255)])
@pytest.mark.parametrize('separate', [True, False])
def test_semantic_label_routes_every_tool_to_mask(tmp_path, kind, label, category, separate):
    result = run(tmp_path, region(kind, label, separate))
    assert result['points'] == []  # Keep submitted-empty particle convention.
    mask = np.asarray(Image.open(tmp_path / result['semantic_mask']))
    assert mask.shape == (80, 100)
    assert mask[30, 20] == category
    assert mask[70, 90] == 255
    assert result['semantic_classes'] == ['Carbon', 'Film', 'Vacuum']


@pytest.mark.parametrize('kind', ['ellipse', 'rectangle', 'polygon', 'brush', 'magicwand'])
@pytest.mark.parametrize('separate', [True, False])
def test_particle_label_routes_every_tool_to_circle(tmp_path, kind, separate):
    result = run(tmp_path, region(kind, 'nanoparticle', separate))
    assert 'semantic_mask' not in result
    point, = result['points']
    assert point[:2] == pytest.approx([20, 30])
    assert point[2] == pytest.approx(10 if kind == 'ellipse' else np.sqrt(400 / np.pi))


def test_shared_gallery_duplicate_is_one_particle(tmp_path):
    tags = region('polygon', 'nanoparticle', True)
    duplicate = copy.deepcopy(tags)
    for tag in duplicate:
        tag['item_index'] = 1
    assert len(run(tmp_path, tags + duplicate)['points']) == 1
    duplicate[0]['id'] = duplicate[1]['id'] = 'different'
    assert len(run(tmp_path, tags + duplicate)['points']) == 2


@pytest.mark.parametrize('case', ['unknown', 'unlabeled', 'wrong_frame', 'wrong_target', 'label_frame', 'label_target', 'ambiguous', 'bad_size', 'rotation', 'empty_brush'])
def test_ambiguous_or_invalid_regions_fail_instead_of_becoming_particles(tmp_path, case):
    tags = region('brush', 'Carbon', True)
    if case == 'unknown': tags[0]['value']['labels'] = ['Unknown']
    if case == 'unlabeled': tags = tags[1:]
    if case == 'wrong_frame':
        for t in tags: t['item_index'] = 2
    if case == 'wrong_target':
        for t in tags: t['to_name'] = 'other'
    if case == 'label_frame': tags[0]['item_index'] = 1
    if case == 'label_target': tags[0]['to_name'] = 'other'
    if case == 'ambiguous': tags[0]['value']['labels'] = ['Carbon', 'nanoparticle']
    if case == 'bad_size': tags[1]['original_width'] = 0
    if case == 'rotation': tags[1]['image_rotation'] = 90
    if case == 'empty_brush': tags[1]['value']['rle'] = mask2rle(np.zeros((80, 100), dtype=np.uint8))
    with pytest.raises(ValueError):
        run(tmp_path, tags)


def test_real_magicwand_geometry_and_labeled_partner_export_once(tmp_path):
    tags = json.loads((ROOT / 'tests/fixtures/ls-1.23-magicwand.json').read_text())
    wand = next(t for t in tags if t['type'] == 'magicwand')
    pair = [t for t in tags if t['id'] == wand['id']]
    result = run(tmp_path, pair)
    assert result['points'] == []
    assert np.all(np.asarray(Image.open(tmp_path / result['semantic_mask'])) == 0)
    for tag in pair:
        if tag['type'] == 'brushlabels':
            tag['value']['brushlabels'] = ['nanoparticle']
    result = run(tmp_path, pair)
    assert len(result['points']) == 1
    assert 'semantic_mask' not in result
    with pytest.raises(ValueError, match='label'):
        run(tmp_path, [wand])


def test_semantic_overlap_ignore_and_order_independence(tmp_path):
    carbon = region('polygon', 'Carbon')
    film = region('brush', 'Film')
    film[0]['id'] = 'film'
    ignored = region('ellipse', 'Ignore')
    ignored[0]['id'] = 'ignore'
    for tags in (carbon + film + ignored, ignored + film + carbon):
        out = run(tmp_path, tags)
        assert np.all(np.asarray(Image.open(tmp_path / out['semantic_mask'])) == 255)


def test_aliases_and_category_ids_are_preserved(tmp_path):
    config = copy.deepcopy(CONFIG)
    config.find(".//Labels[@name='semantic']/Label[@value='Film']").set('alias', 'f')
    tags = region('polygon', 'f', True)
    out = run(tmp_path, tags, config)
    assert np.asarray(Image.open(tmp_path / out['semantic_mask']))[30, 20] == 1
    assert out['semantic_classes'] == ['Carbon', 'Film', 'Vacuum']


def test_ui_uses_separate_labels_and_unique_shared_tools():
    root = ET.fromstring(yaml.safe_load((ROOT / 'config.yml').read_text())['config'])
    assert root.find(".//Labels[@name='semantic']") is not None
    assert root.find(".//Labels[@name='labels']") is not None
    for kind in ('Ellipse', 'Polygon', 'Brush', 'Magicwand'):
        assert len(root.findall('.//' + kind)) == 1
    assert root.find('.//PolygonLabels') is None
    assert [node.get('category') for node in root.findall(".//Labels[@name='semantic']/Label")] == ['0', '1', '2', '255']
    hotkeys = [node.get('hotkey') for node in root.findall('.//Label')]
    assert hotkeys == ['p', 'c', 'f', 'v', 'i']
    assert [panel.get('value') for panel in root.findall('.//Panel')] == [
        'Split', 'Particle instances', 'Segmentation', 'Smart tools']
    particle = root.find(".//Panel[@value='Particle instances']")
    semantic = root.find(".//Panel[@value='Segmentation']")
    assert particle.find('Ellipse').attrib == {'name': 'points', 'toName': 'image'}
    assert [node.tag for node in semantic if node.tag in ('Polygon', 'Brush', 'Magicwand')] == [
        'Polygon', 'Brush', 'Magicwand']
    assert 'Drawing tools' not in ET.tostring(root, encoding='unicode')


@pytest.mark.parametrize('kind', ['ellipse', 'polygon', 'brush', 'magicwand'])
def test_real_toolbox_export_manifest_loads_joint_targets(tmp_path, monkeypatch, kind):
    """Only SDK transport is stubbed; host export, files and dataset are real."""
    import os
    import runpy
    import sys
    from types import SimpleNamespace
    import label_studio_sdk
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.stem_tasks import TaskConfig

    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    for name, value in [('BF.png', 17), ('HAADF.png', 91)]:
        Image.new('L', (100, 80), value).save(dataset / name)
    tags = region(kind, 'Carbon', True)
    particle = region(kind, 'nanoparticle', True)
    for tag in particle:
        tag['id'] = 'particle'
    payload = json.dumps([dict(data={'images': [
        f'https://example/data/local-files/?d=dataset/{name}' for name in ('BF.png', 'HAADF.png')]},
        annotations=[dict(result=tags + particle)], split='Train')]).encode()
    exports = SimpleNamespace(create=lambda **kw: SimpleNamespace(status='completed', id=123),
                              download=lambda **kw: [payload])
    xml = yaml.safe_load((ROOT / 'config.yml').read_text())['config']
    monkeypatch.setattr(label_studio_sdk, 'LabelStudio', lambda **kw: SimpleNamespace(
        projects=SimpleNamespace(exports=exports, get=lambda **kw: SimpleNamespace(label_config=xml))))
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
    host = ROOT.parents[1] / 'toolbox'
    runpy.run_path(str(host / 'apps/ls-utils/export.py'), run_name='__main__')
    manifest = tmp_path / 'export/manifest.json'
    data = json.loads(manifest.read_text())
    assert data['version'] == 4
    point, = data['train'][0]['points']
    assert point[:2] == pytest.approx([20, 30])
    sample = ToolboxDataset(manifest, TaskConfig(True, True), size=(100, 80))[0]
    assert sample['image'][:, 0, 0].tolist() == [17, 91, 0]
    assert sample['shape_coef'][0, 30, 20] == pytest.approx(point[2])
    assert sample['semantic_segmentation'][30, 20] == 0
    assert sample['semantic_segmentation'][70, 90] == 255
