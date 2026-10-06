"""Default weight selection at the schema, serving and training boundaries."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_schema_has_nonempty_default_weight_values():
    properties = json.loads((ROOT / 'model.json').read_text())['properties']
    for key in ('model', 'localisation'):
        assert properties[key]['default'] == 'default'
    # Toolbox checks explicit value presence, not schema defaults. The runtime
    # enforces task-specific checkpoint requirements instead.
    assert 'infer' not in properties['model'].get('required', [])
    assert properties['model']['format'] == 'file:checkpoint.pth'


@pytest.mark.parametrize('value', [None, '', 'default'])
def test_default_weight_values_resolve_to_no_override(value):
    from stem_plugin.checkpoint import checkpoint_override
    assert checkpoint_override(value) is None


@pytest.mark.parametrize('value', ['/tmp/checkpoint.pth', '/tmp/default', 'DEFAULT'])
def test_checkpoint_paths_are_preserved(value):
    from stem_plugin.checkpoint import checkpoint_override
    assert checkpoint_override(value) == value


@pytest.fixture
def serving(monkeypatch):
    from stem_plugin import serving
    calls = []
    state = {'backbone': 'resnet18', 'center_model_state_dict': {'stock': True}}

    def download(url, **kwargs):
        calls.append(('download', url, kwargs))
        return state

    def load(path, **kwargs):
        calls.append(('file', path, kwargs))
        return dict(state, center_model_state_dict={'custom': True})

    class Runtime:
        def __init__(self, tasks, device, backbone, pretrained):
            self.tasks = tasks
        def load(self, checkpoint, **kwargs):
            calls.append(('runtime', checkpoint, kwargs))
        def eval(self):
            return self

    monkeypatch.setattr(serving.torch.hub, 'load_state_dict_from_url', download)
    monkeypatch.setattr(serving, 'safe_torch_load', load)
    monkeypatch.setattr(serving, 'StemRuntime', Runtime)
    return serving, calls


@pytest.mark.parametrize('value', [None, '', 'default'])
def test_particle_inference_downloads_default_weights(serving, value):
    module, calls = serving
    module.load_runtime({'model': value, 'localisation': value}, 'cpu')
    assert calls[0] == ('download', 'https://data.vicos.si/skokec/STEM/checkpoint.pth', {'map_location': 'cpu'})
    assert [call[0] for call in calls] == ['download', 'runtime']
    assert calls[-1][1]['center_model_state_dict'] == {'stock': True}


@pytest.mark.parametrize('particles', [False, True])
@pytest.mark.parametrize('value', [None, '', 'default'])
def test_segmentation_still_requires_trained_weights(serving, particles, value):
    module, calls = serving
    with pytest.raises(ValueError, match='segmentation inference requires trained semantic weights'):
        module.load_runtime({'model': value, 'nanoparticles': particles, 'segmentation': True}, 'cpu')
    assert calls == []


def test_custom_inference_weights_and_localizer_are_preserved(serving):
    module, calls = serving
    module.load_runtime({'model': '/tmp/model.pth', 'localisation': '/tmp/localizer.pth'}, 'cpu')
    assert [(call[0], call[1]) for call in calls[:2]] == [('file', '/tmp/model.pth'), ('file', '/tmp/localizer.pth')]
    assert calls[-1][1]['center_model_state_dict'] == {'custom': True}


def test_custom_inference_weights_with_default_localizer_keep_embedded_state(serving):
    module, calls = serving
    module.load_runtime({'model': '/tmp/model.pth', 'localisation': 'default'}, 'cpu')
    assert [call[0] for call in calls] == ['file', 'runtime']
    assert calls[-1][1]['center_model_state_dict'] == {'custom': True}


@pytest.mark.parametrize('particles,model,localisation,expected', [
    (True, None, None, [('center', '/cache/cedirnet-stem/localization_checkpoint.pth')]),
    (True, 'default', 'default', [('center', '/cache/cedirnet-stem/localization_checkpoint.pth')]),
    (True, '/tmp/model.pth', 'default', [('model', '/tmp/model.pth')]),
    (True, 'default', '/tmp/localizer.pth', [('center', '/tmp/localizer.pth')]),
    (True, '/tmp/model.pth', '/tmp/localizer.pth', [('model', '/tmp/model.pth'), ('center', '/tmp/localizer.pth')]),
    (False, 'default', 'default', []),
    (False, 'default', '/missing/localizer.pth', []),
])
def test_training_default_preserves_initialization_policy(monkeypatch, particles, model, localisation, expected):
    import train
    calls = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass
        def load(self, state):
            calls.append(('model', state))
        def load_center(self, state):
            calls.append(('center', state))

    class DatasetReached(Exception):
        pass

    def dataset(*args, **kwargs):
        raise DatasetReached

    monkeypatch.setenv('TOOLBOX_CACHE', '/cache')
    monkeypatch.setattr(train, 'StemRuntime', Runtime)
    monkeypatch.setattr(train, 'safe_torch_load', lambda path, **kwargs: path)
    monkeypatch.setattr(train, 'ToolboxDataset', dataset)
    trainer = train.Trainer({'model': model, 'localisation': localisation,
                             'nanoparticles': particles, 'segmentation': not particles,
                             'device': 'cpu', 'manifest': '/tmp/manifest.json'})
    with pytest.raises(DatasetReached):
        trainer.initialize()
    assert calls == expected


def test_modelargs_supplies_default_and_preserves_custom_paths(monkeypatch):
    import modelargs
    monkeypatch.setattr(sys, 'argv', ['infer', '--'])
    values = modelargs.parse(str(ROOT / 'model.json'))
    assert values['model'] == values['localisation'] == 'default'
    monkeypatch.setattr(sys, 'argv', ['infer', '--', '--model', '/tmp/custom.pth'])
    assert modelargs.parse(str(ROOT / 'model.json'))['model'] == '/tmp/custom.pth'
