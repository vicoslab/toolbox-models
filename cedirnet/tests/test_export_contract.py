"""Both counting adapters accept flat and Toolbox-nested result lists."""
import importlib.util
import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
from label_studio_sdk.converter.brush import mask2rle

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('model', ['cedirnet', 'cedirnet-stem'])
def test_unassigned_manifest_defaults_to_training_only(model):
    """Execute the actual loader's split selection, without ML dependencies."""
    sample = {'image_path': 'image.png', 'points': [[25, 40]]}
    manifest = {'version': 4, 'data': [sample]}
    if model == 'cedirnet':
        patch = (ROOT / model / '0002-Add-GenericDataset.patch').read_text()
        hunk = patch.split('@@ -0,0 +1,', 1)[1].split('\n', 1)[1].split('diff --git', 1)[0]
        source = '\n'.join(line[1:] for line in hunk.splitlines() if line.startswith('+'))
        tree = ast.parse(source)
        selection = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'split in items')
        initial = 'items'
    else:
        tree = ast.parse((ROOT / model / 'stem_plugin/toolbox_dataset.py').read_text())
        selection = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'items' for t in node.targets))
        initial = 'data'
    code = compile(ast.Module(body=[selection], type_ignores=[]), model, 'exec')
    for split in ('train', 'val', 'test'):
        namespace = {initial: manifest.copy(), 'split': split}
        exec(code, namespace)
        assert namespace['items'] == ([sample] if split == 'train' else [])


@pytest.fixture(params=['cedirnet', 'cedirnet-stem'])
def adapter(request, monkeypatch):
    directory = ROOT / request.param
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('contract_' + request.param, directory / 'ls_adapter.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return request.param, module


def export(adapter, tmp_path, annotations):
    name, module = adapter
    config = ET.fromstring('<View><Label value="Carbon" category="0"/></View>')
    paths = ['BF.png', 'HAADF.png'] if name == 'cedirnet-stem' else ['image.png']
    return module.export(annotations, tmp_path, paths, False, config)


def region(kind='vector'):
    return dict(id='particle', type=kind, original_width=100, original_height=80,
                value={'vertices': [{'x': 25, 'y': 50}, {'x': 50, 'y': 75}]})


@pytest.mark.parametrize('nested', [False, True])
def test_choices_and_geometry(adapter, tmp_path, nested):
    tags = [dict(type='choices', value={'choices': ['Train']}), region()]
    result = export(adapter, tmp_path, [tags] if nested else tags)
    assert result['points'] == [[25, 40, 50, 60]]


def test_submitted_empty(adapter, tmp_path):
    assert export(adapter, tmp_path, [[]]) == {'points': []}


@pytest.mark.parametrize('nested', [False, True])
def test_missing_split_does_not_require_choice_control(adapter, tmp_path, nested):
    tags = [region()]
    without_split = export(adapter, tmp_path, [tags] if nested else tags)
    for split in ('Train', 'Validation', 'Test', 'Auto'):
        with_choice = tags + [dict(type='choices', value={'choices': [split]})]
        assert export(adapter, tmp_path, [with_choice] if nested else with_choice) == without_split
    assert without_split['points'] == [[25, 40, 50, 60]]


@pytest.mark.parametrize('annotations', [None, []])
def test_missing_submission(adapter, tmp_path, annotations):
    assert not export(adapter, tmp_path, annotations)


@pytest.mark.parametrize('annotations', [[[region()], [region()]], [[[region()]]], [region(), []]])
def test_ambiguous_or_malformed_nesting(adapter, tmp_path, annotations):
    with pytest.raises(ValueError, match='annotation|result'):
        export(adapter, tmp_path, annotations)


@pytest.mark.parametrize('kind', ['brushlabels', 'brush', 'magicwand'])
def test_brush_geometry(adapter, tmp_path, kind):
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:40, 10:30] = 255
    tag = region(kind)
    tag['value'] = {'rle': mask2rle(mask)}
    result = export(adapter, tmp_path, [[tag]])
    assert result['points'][0][:2] == [19, 29]


def test_stem_semantic_brush(adapter, tmp_path):
    if adapter[0] != 'cedirnet-stem':
        pytest.skip('semantic masks are STEM-only')
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[:, :30] = 255
    tag = region('brushlabels')
    tag['value'] = {'brushlabels': ['Carbon'], 'rle': mask2rle(mask)}
    result = export(adapter, tmp_path, [[tag]])
    decoded = np.asarray(Image.open(tmp_path / result['semantic_mask']))
    assert np.all(decoded[:, :30] == 0)
    assert np.all(decoded[:, 30:] == 255)
    assert result['semantic_classes'] == ['Carbon']
