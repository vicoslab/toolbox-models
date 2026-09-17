"""Strict localization compatibility; optional released-checkpoint integration."""
import os

import pytest
import torch

from stem_plugin.runtime import StemRuntime

LEGACY = (
    'module.instance_mask_estimator.xym_1024',
    'module.center_augmentator.xym',
    'module.instance_center_estimator.kernel_cos',
    'module.instance_center_estimator.kernel_sin',
)
FIRST = 'module.instance_center_estimator.conv_start.0.weight'
GAUSSIAN = 'module.instance_center_estimator.gaussian_blur.conv.weight'


@pytest.fixture
def runtime():
    from models import get_center_model
    from stem_plugin.base_config import get_args, NUM_VECTOR_FIELDS
    torch.set_num_threads(2)
    runtime = StemRuntime.__new__(StemRuntime)
    torch.nn.Module.__init__(runtime)
    args = get_args()['center_model']
    center = get_center_model(args['name'], args['kwargs'], is_learnable=True)
    center.init_output(NUM_VECTOR_FIELDS)
    runtime.center_model = torch.nn.DataParallel(center).requires_grad_(False)
    return runtime


def legacy_state(runtime):
    values = {k: v.clone() for k, v in runtime.center_model.state_dict().items()}
    values.update({k: torch.zeros(1) for k in LEGACY})
    values[FIRST] = torch.cat([values[FIRST], torch.ones_like(values[FIRST])], dim=1)
    return values


def test_legacy_geometry_loads_without_mutating_checkpoint(runtime, caplog):
    values = legacy_state(runtime)
    runtime.load_center({'center_model_state_dict': values})
    assert all(key in caplog.text for key in LEGACY)
    actual = runtime.center_model.state_dict()
    for key, value in actual.items():
        assert torch.equal(value, values[key][:, :2] if key == FIRST else values[key])
    assert all(key in values for key in LEGACY)
    assert values[FIRST].shape[1] == 4


def test_missing_fixed_gaussian_retains_runtime_value(runtime):
    values = legacy_state(runtime)
    values.pop(GAUSSIAN)
    # Like main's strict=False loader, missing geometry leaves runtime state alone.
    runtime.center_model.state_dict()[GAUSSIAN].fill_(42)
    before = runtime.center_model.state_dict()[GAUSSIAN].clone()
    runtime.load_center({'center_model_state_dict': values})
    assert torch.equal(runtime.center_model.state_dict()[GAUSSIAN], before)
    assert GAUSSIAN not in values


def test_official_style_load_preserves_initialized_gaussian(runtime):
    values = legacy_state(runtime)
    initialized = values.pop(GAUSSIAN)
    runtime.load_center({'center_model_state_dict': values})
    assert torch.equal(runtime.center_model.state_dict()[GAUSSIAN], initialized)


@pytest.mark.parametrize('key', ['module.instance_center_estimator.learned.weight',
                                'module.instance_mask_estimator.weight',
                                'module.instance_center_estimator.kernel_cos_extra'])
def test_unexpected_keys_still_fail(runtime, key):
    values = legacy_state(runtime)
    values[key] = torch.ones(1)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        runtime.load_center({'center_model_state_dict': values})


def test_missing_learned_key_still_fails(runtime):
    values = legacy_state(runtime)
    del values[FIRST]
    with pytest.raises(RuntimeError, match='Missing key'):
        runtime.load_center({'center_model_state_dict': values})


@pytest.mark.parametrize('shape', [(16, 3, 3, 3), (16, 5, 3, 3), (17, 4, 3, 3), (16, 4, 5, 5)])
def test_unrecognized_first_layer_shape_still_fails(runtime, shape):
    values = legacy_state(runtime)
    values[FIRST] = torch.zeros(shape)
    with pytest.raises(RuntimeError, match='size mismatch'):
        runtime.load_center({'center_model_state_dict': values})


def test_expected_geometry_is_not_discarded(runtime):
    estimator = runtime.center_model.module.instance_center_estimator
    estimator.register_buffer('kernel_cos', torch.zeros(1))
    values = legacy_state(runtime)
    values['module.instance_center_estimator.kernel_cos'] = torch.ones(1)
    runtime.load_center({'center_model_state_dict': values})
    assert torch.equal(estimator.kernel_cos, torch.ones(1))


@pytest.mark.parametrize('key', [GAUSSIAN, 'module.instance_center_estimator.conv_end.0.weight'])
def test_same_name_shape_mismatch_still_fails(runtime, key):
    values = legacy_state(runtime)
    values[key] = torch.zeros(1)
    with pytest.raises(RuntimeError, match='size mismatch'):
        runtime.load_center({'center_model_state_dict': values})


def test_current_checkpoint_roundtrip(runtime):
    values = runtime.center_model.state_dict()
    runtime.load_center({'center_model_state_dict': values})
    assert all(torch.equal(v, values[k]) for k, v in runtime.center_model.state_dict().items())


@pytest.mark.parametrize('particles,semantic', [(True, False), (True, True), (False, True)])
def test_default_localization_training(tmp_path, monkeypatch, particles, semantic):
    from test_tasks import fixture_manifest
    from train import Trainer
    from stem_plugin.checkpoint import safe_torch_load
    from stem_plugin.stem_tasks import TaskConfig
    checkpoint = os.getenv('CEDIRNET_STEM_TEST_LOCALIZATION')
    if particles and not checkpoint:
        pytest.skip('set CEDIRNET_STEM_TEST_LOCALIZATION to the official localization checkpoint')
    torch.set_num_threads(2)
    monkeypatch.setenv('TOOLBOX_CACHE', str(tmp_path/'cache'))
    if particles:
        target = tmp_path/'cache'/'cedirnet-stem'/'localization_checkpoint.pth'
        target.parent.mkdir(parents=True)
        target.symlink_to(os.path.abspath(checkpoint))
    options = dict(manifest=str(fixture_manifest(tmp_path, particles)),
                   nanoparticles=particles, segmentation=semantic, width=64, height=64,
                   batch_size=1, workers=0, epochs=1, backbone='resnet18', device='cpu')
    trainer = Trainer(options)
    trainer.initialize()  # No model/localisation override: exercise the default path.
    if particles:
        official = safe_torch_load(checkpoint, map_location='cpu')['center_model_state_dict']
        for key, value in trainer.runtime.center_model.state_dict().items():
            if key != GAUSSIAN:
                assert torch.equal(value, official[key][:, :2] if key == FIRST else official[key])
    else:
        assert trainer.runtime.center_model is None
    before = next(trainer.runtime.parameters()).detach().clone()
    trainer.runtime.train()
    loss, parts = trainer.runtime.loss(next(iter(trainer.loaders['train'])))
    assert torch.isfinite(loss)
    loss.backward()
    trainer.optimizer.step()
    assert not torch.equal(before, next(trainer.runtime.parameters()).detach())
    assert set(parts) == ({'particle_loss'} if particles else set()) | ({'semantic_loss'} if semantic else set())
    path = tmp_path/'trained.pth'
    torch.save(trainer.runtime.checkpoint(0), path)
    clone = StemRuntime(TaskConfig(particles, semantic), backbone='resnet18')
    clone.load(safe_torch_load(path, map_location='cpu'), inference=True)
