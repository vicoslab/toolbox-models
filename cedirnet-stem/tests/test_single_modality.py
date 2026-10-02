"""Explicit detector availability; paired annotation remains unchanged."""
import base64
import importlib.util
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage

from stem_plugin.annotations import load_stem_image
from stem_plugin.serving import load_pairs


ROOT = Path(__file__).resolve().parents[1]
VALUES = np.array([[0, 17, 91], [255, 128, 3]], dtype=np.uint8)


def image_bytes(values=VALUES, *, format='PNG', exif=None):
    stream = io.BytesIO()
    options = {'exif': exif} if exif is not None else {}
    Image.fromarray(values).save(stream, format=format, **options)
    stream.seek(0)
    return stream


@pytest.mark.parametrize('modality,channel', [('BF', 0), ('HAADF', 1)])
@pytest.mark.parametrize('filename', ['upload.png', 'misleading_HAADF_BF.png'])
def test_single_image_has_fixed_raw_channels(modality, channel, filename):
    upload = FileStorage(stream=image_bytes(), filename=filename)
    images = load_pairs([upload], modality=modality)
    assert len(images) == 1
    image = images[0]
    assert image.dtype == np.uint8
    assert image.shape == (2, 3, 3)
    np.testing.assert_array_equal(image[:, :, channel], VALUES)
    assert not image[:, :, 1-channel].any()
    assert not image[:, :, 2].any()


@pytest.mark.parametrize('modality,channel', [('BF', 0), ('HAADF', 1)])
@pytest.mark.parametrize('format', ['PNG', 'TIFF'])
def test_single_image_uses_existing_grayscale_conversion(modality, channel, format):
    # Deliberately preserve Pillow's current conversion, not new uint16 scaling.
    values = np.array([[0, 255, 256], [1024, 32768, 65535]], dtype=np.uint16)
    source = image_bytes(values, format=format)
    with Image.open(source) as original:
        expected = np.asarray(ImageOps.exif_transpose(original).convert('L'))
    source.seek(0)
    image = load_pairs([source], modality=modality)[0]
    np.testing.assert_array_equal(image[:, :, channel], expected)


@pytest.mark.parametrize('modality,channel', [('BF', 0), ('HAADF', 1)])
def test_single_image_applies_exif_orientation(modality, channel):
    exif = Image.Exif()
    exif[274] = 6
    image = load_pairs([image_bytes(exif=exif)], modality=modality)[0]
    assert image.shape == (3, 2, 3)
    np.testing.assert_array_equal(image[:, :, channel], np.rot90(VALUES, k=-1))


@pytest.mark.parametrize('filename', ['generic.png', 'sample_BF.png', 'sample_HAADF.png'])
def test_single_image_without_mode_never_guesses_from_filename(tmp_path, filename):
    path = tmp_path / filename
    Image.fromarray(VALUES).save(path)
    # Even a real inferred pair must not override the upload's missing selection.
    for name in ['sample_BF.png', 'sample_HAADF.png']:
        Image.fromarray(VALUES).save(tmp_path / name)
    with pytest.raises(ValueError, match=r'(?s)(BF.*HAADF|HAADF.*BF)') as error:
        load_pairs([path])
    assert 'modality' in str(error.value).lower()


@pytest.mark.parametrize('modality', ['', 'bf', 'haadf', 'RGB', 'auto', 'BF+HAADF'])
def test_invalid_modality_is_rejected(modality):
    with pytest.raises(ValueError, match='modality'):
        load_pairs([image_bytes()], modality=modality)


@pytest.mark.parametrize('modality', [None, 'paired', 'BF', 'HAADF'])
def test_no_images_is_rejected(modality):
    with pytest.raises(ValueError, match='image'):
        load_pairs([], modality=modality)


@pytest.mark.parametrize('modality', ['BF', 'HAADF'])
@pytest.mark.parametrize('count', [2, 3, 4])
def test_single_modality_requires_exactly_one_file(modality, count):
    with pytest.raises(ValueError, match='exactly one'):
        load_pairs([image_bytes() for _ in range(count)], modality=modality)


@pytest.mark.parametrize('modality', [None, 'paired'])
@pytest.mark.parametrize('count', [2, 4])
def test_paired_batches_preserve_channel_order(modality, count):
    files = [image_bytes(VALUES+i) for i in range(count)]
    images = load_pairs(files, modality=modality)
    assert len(images) == count//2
    for index, image in enumerate(images):
        np.testing.assert_array_equal(image[:, :, 0], VALUES+index*2)
        np.testing.assert_array_equal(image[:, :, 1], VALUES+index*2+1)
        assert not image[:, :, 2].any()


@pytest.mark.parametrize('modality', [None, 'paired'])
def test_odd_paired_batch_is_rejected(modality):
    with pytest.raises(ValueError, match='pairs'):
        load_pairs([image_bytes() for _ in range(3)], modality=modality)


def test_paired_dimension_mismatch_stays_an_error():
    with pytest.raises(ValueError, match='dimensions differ'):
        load_pairs([image_bytes(), image_bytes(np.ones((3, 2), dtype=np.uint8))])


def test_paired_loader_retains_legacy_filename_inference(tmp_path):
    bf = tmp_path / 'sample_BF.png'
    haadf = tmp_path / 'sample_HAADF.png'
    Image.fromarray(VALUES).save(bf)
    Image.fromarray(VALUES+1).save(haadf)
    for source in [bf, haadf]:
        image = np.asarray(load_stem_image(source))
        np.testing.assert_array_equal(image[:, :, 0], VALUES)
        np.testing.assert_array_equal(image[:, :, 1], VALUES+1)
    with pytest.raises(ValueError, match='swapped'):
        load_stem_image(haadf, bf)
    haadf.unlink()
    with pytest.raises(FileNotFoundError, match='HAADF pair missing'):
        load_stem_image(bf)


@pytest.mark.parametrize('modality,channel', [('BF', 0), ('HAADF', 1)])
@pytest.mark.parametrize('particles,semantic', [(True, False), (False, True), (True, True)])
def test_http_singleton_reaches_real_task_models(infer_module, monkeypatch, modality, channel, particles, semantic):
    import torch
    from stem_plugin.runtime import StemRuntime
    from stem_plugin.stem_tasks import TaskConfig

    torch.set_num_threads(2)
    module, _ = infer_module
    tasks = TaskConfig(particles, semantic)
    # Real public upstream FPN/localizer, no download, random resnet18 weights.
    # This proves plumbing, not released-checkpoint accuracy or robustness.
    runtime = StemRuntime(tasks, device='cpu', backbone='resnet18', pretrained=False).eval()
    monkeypatch.setattr(module, 'RUNTIME', runtime)
    inputs, outputs, hooks = {}, {}, []
    def capture_input(name):
        return lambda model, args: inputs.__setitem__(name, args[0].detach().cpu().clone())
    def capture_output(name):
        return lambda model, args, output: outputs.__setitem__(name, output.detach().cpu())
    for name, model in [('particles', runtime.particle_model), ('semantic', runtime.semantic_model)]:
        if model is not None:
            hooks.append(model.register_forward_pre_hook(capture_input(name)))
            hooks.append(model.register_forward_hook(capture_output(name)))
    try:
        response = module.app.test_client().post('/infer', data={
            'modality': modality, 'images': (image_bytes(), 'generic.tif')})
    finally:
        for hook in hooks:
            hook.remove()
    assert response.status_code == 200, response.json
    assert response.json['input_mode'] == modality
    assert response.json['tasks'] == tasks.to_dict()
    assert len(inputs) == int(particles)+int(semantic)
    expected = np.asarray(Image.fromarray(VALUES).resize((64, 64), Image.Resampling.BILINEAR))
    for tensor in inputs.values():
        assert tensor.dtype == torch.float32
        assert tuple(tensor.shape) == (1, 3, 64, 64)
        np.testing.assert_array_equal(tensor[0, channel].numpy(), expected)
        assert not tensor[0, 1-channel].any()
        assert not tensor[0, 2].any()
    assert all(torch.isfinite(output).all() for output in outputs.values())
    if semantic:
        mask = response.json['segmentation'][0]
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(mask['mask_png'].split(',', 1)[1]))))
        assert decoded.shape == VALUES.shape
        assert set(np.unique(decoded)).issubset(range(len(tasks.classes)))


class RuntimeProbe:
    """Replace only eager checkpoint initialization in entrypoint tests."""
    def __init__(self):
        self.calls = []

    def predict(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return {'centers': [[] for _ in images], 'scores': [[] for _ in images],
                'radii': [[] for _ in images], 'segmentation': [None for _ in images],
                'tasks': {'nanoparticles': False, 'segmentation': False}}


@pytest.fixture
def infer_module(tmp_path, monkeypatch):
    import stem_plugin.serving as serving
    probe = RuntimeProbe()
    monkeypatch.setattr(serving, 'load_runtime', lambda options, device: probe)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', ['infer', '--width', '64', '--height', '64'])
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'backend-cache'))
    import label_studio_ml.api
    import label_studio_ml.model
    from label_studio_ml.cache import SqliteCache
    monkeypatch.setattr(label_studio_ml.model, 'CACHE', SqliteCache(str(tmp_path / 'backend-cache')))
    importlib.reload(label_studio_ml.api)
    spec = importlib.util.spec_from_file_location('single_modality_infer', ROOT / 'infer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, probe


@pytest.mark.parametrize('modality,channel', [('BF', 0), ('HAADF', 1)])
def test_http_single_upload_forwards_explicit_modality(infer_module, modality, channel):
    module, probe = infer_module
    response = module.app.test_client().post('/infer', data={
        'modality': modality, 'images': (image_bytes(), 'arbitrary.png')})
    assert response.status_code == 200, response.json
    assert response.json['input_mode'] == modality
    images, options = probe.calls[-1]
    assert options['score_threshold'] == 0
    np.testing.assert_array_equal(images[0][:, :, channel], VALUES)
    assert not images[0][:, :, 1-channel].any()
    assert not images[0][:, :, 2].any()


@pytest.mark.parametrize('modality,count,match', [
    (None, 1, 'modality'), ('paired', 1, 'modality'), ('RGB', 1, 'modality'),
    ('', 1, 'modality'), ('BF', 2, 'exactly one'), ('HAADF', 3, 'exactly one'),
    (None, 0, 'image'), ('BF', 0, 'image'), ('HAADF', 0, 'image'),
    (None, 3, 'pairs'), ('paired', 3, 'pairs'),
])
def test_http_rejects_invalid_requests_before_prediction(infer_module, modality, count, match):
    module, probe = infer_module
    data = {'images': [(image_bytes(), f'upload-{i}.png') for i in range(count)]}
    if modality is not None:
        data['modality'] = modality
    response = module.app.test_client().post('/infer', data=data)
    assert response.status_code == 400
    assert match in response.json['error'].lower()
    assert probe.calls == []


@pytest.mark.parametrize('modality', [None, 'paired'])
def test_http_paired_batches_remain_supported(infer_module, modality):
    module, probe = infer_module
    data = {'images': [(image_bytes(VALUES+i), f'upload-{i}.png') for i in range(4)]}
    if modality is not None:
        data['modality'] = modality
    response = module.app.test_client().post('/infer', data=data)
    assert response.status_code == 200, response.json
    assert response.json['input_mode'] == 'paired'
    assert len(probe.calls[-1][0]) == 2


def test_http_dimension_mismatch_remains_a_400(infer_module):
    module, probe = infer_module
    response = module.app.test_client().post('/infer', data={
        'images': [(image_bytes(), 'one.png'),
                   (image_bytes(np.ones((3, 2), dtype=np.uint8)), 'two.png')]})
    assert response.status_code == 400
    assert 'dimensions differ' in response.json['error']
    assert probe.calls == []


@pytest.mark.parametrize('data', [{'image': 'only.png', 'modality': 'BF'},
                                 {'images': ['only.png'], 'modality': 'HAADF'},
                                 {'images': ['a.png', 'b.png', 'c.png']}])
def test_preannotation_remains_strictly_paired(infer_module, data):
    module, probe = infer_module
    backend = object.__new__(module.CeDiRNetSTEM)
    with pytest.raises(ValueError, match=r'data.images \[BF, HAADF\]'):
        backend.predict([{'data': data}])
    assert probe.calls == []


def test_preannotation_paired_path_ignores_singleton_metadata(infer_module, tmp_path):
    module, probe = infer_module
    paths = [tmp_path / f'image-{i}.png' for i in range(2)]
    for index, path in enumerate(paths):
        Image.fromarray(VALUES+index).save(path)
    backend = module.CeDiRNetSTEM(
        project_id='paired-single-modality-regression',
        label_config='<View><Image name="image" value="$images"/></View>')
    backend.get_local_path = lambda path, task_id=None: path
    result = backend.predict([{'id': 1, 'data': {'images': list(map(str, paths)), 'modality': 'BF'}}])
    assert len(result.predictions) == 1
    images, options = probe.calls[-1]
    np.testing.assert_array_equal(images[0][:, :, 0], VALUES)
    np.testing.assert_array_equal(images[0][:, :, 1], VALUES+1)
    assert options['score_threshold'] == module.CMD_ARGS['score_threshold']


def test_paired_preannotations_preserve_current_particle_and_semantic_controls(infer_module, tmp_path, monkeypatch):
    import yaml
    from label_studio_converter.brush import decode_rle
    from stem_plugin.semantic_results import encode_mask

    module, probe = infer_module
    paths = [tmp_path / f'image-{i}.png' for i in range(2)]
    for index, path in enumerate(paths):
        Image.fromarray(VALUES+index).save(path)
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    def fixture_prediction(images, **options):
        probe.calls.append((images, options))
        return dict(tasks={'nanoparticles': True, 'segmentation': True},
                    centers=[[[.25, .5]]], radii=[[1.0]], scores=[[.8]],
                    segmentation=[encode_mask(mask, ['Carbon', 'Film', 'Vacuum'])])
    monkeypatch.setattr(probe, 'predict', fixture_prediction)
    backend = module.CeDiRNetSTEM(
        project_id='current-paired-preannotation-config',
        label_config=yaml.safe_load((ROOT / 'config.yml').read_text())['config'])
    backend.get_local_path = lambda path, task_id=None: path
    result = backend.predict([{'data': {'images': list(map(str, paths)), 'modality': 'HAADF'}}])
    annotation = result.predictions[0].model_dump()
    assert annotation['score'] == .8
    regions = annotation['result']
    particle = regions[0]
    assert particle['type'] == 'ellipselabels'
    assert particle['from_name'] == 'labels'
    assert particle['to_name'] == 'image'
    assert particle['original_width'] == 3 and particle['original_height'] == 2
    assert particle['value'] == {'labels': ['PtCo'], 'x': 25.0, 'y': 50.0,
                                 'radiusX': pytest.approx(100/3), 'radiusY': 50.0, 'rotation': 0}
    assert len(regions) == 4
    for index, brush in enumerate(regions[1:]):
        assert brush['type'] == 'brushlabels'
        assert brush['from_name'] == 'semantic'
        assert brush['value']['polygonlabels'] == [['Carbon', 'Film', 'Vacuum'][index]]
        alpha = np.asarray(decode_rle(brush['value']['rle'])).reshape(2, 3, 4)[:, :, 3]
        np.testing.assert_array_equal(alpha, (mask == index).astype(np.uint8)*255)
    images, options = probe.calls[-1]
    np.testing.assert_array_equal(images[0][:, :, 0], VALUES)
    np.testing.assert_array_equal(images[0][:, :, 1], VALUES+1)
    assert options['score_threshold'] == module.CMD_ARGS['score_threshold']


@pytest.mark.parametrize('modality', ['BF', 'HAADF'])
@pytest.mark.parametrize('after_separator', [False, True])
def test_cli_singleton_modality_with_modelargs_separator(tmp_path, monkeypatch, modality, after_separator):
    import stem_plugin.serving as serving
    probe = RuntimeProbe()
    monkeypatch.setattr(serving, 'load_runtime', lambda options, device: probe)
    path = tmp_path / 'arbitrary.png'
    Image.fromarray(VALUES).save(path)
    (tmp_path / 'model.json').write_text((ROOT / 'model.json').read_text())
    monkeypatch.chdir(tmp_path)
    args = [str(ROOT / 'infer.py'), str(path)]
    if not after_separator:
        args += ['--modality', modality]
    args += ['--', '--width', '64', '--height', '64']
    if after_separator:
        args += ['--modality', modality]
    monkeypatch.setattr(sys, 'argv', args)
    runpy.run_path(str(ROOT / 'infer.py'), run_name='__main__')
    response = json.loads((tmp_path / 'result.json').read_text())
    assert response['input_mode'] == modality
    images, options = probe.calls[-1]
    np.testing.assert_array_equal(images[0][:, :, ['BF', 'HAADF'].index(modality)], VALUES)
    assert options['size'] == (64, 64)


def test_cli_legacy_two_positional_images(tmp_path, monkeypatch):
    import stem_plugin.serving as serving
    probe = RuntimeProbe()
    monkeypatch.setattr(serving, 'load_runtime', lambda options, device: probe)
    paths = [tmp_path / f'image-{i}.png' for i in range(2)]
    for index, path in enumerate(paths):
        Image.fromarray(VALUES+index).save(path)
    (tmp_path / 'model.json').write_text((ROOT / 'model.json').read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(ROOT / 'infer.py'), *map(str, paths), '--', '--width', '64'])
    runpy.run_path(str(ROOT / 'infer.py'), run_name='__main__')
    images, _ = probe.calls[-1]
    np.testing.assert_array_equal(images[0][:, :, 0], VALUES)
    np.testing.assert_array_equal(images[0][:, :, 1], VALUES+1)
