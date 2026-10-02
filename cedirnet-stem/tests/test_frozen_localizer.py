"""Real STEM training must freeze localizer buffers, not decode loss inputs.

Run with upstream STEM src on PYTHONPATH. CEDIRNET_STEM_TEST_LOCALIZATION
optionally exercises the released pretrained statistics instead of initialized
ones; no model, groundtruth, loss, or localization stand-ins are used.
"""
import copy
import os

import numpy as np
import pytest
import torch

from stem_plugin.checkpoint import safe_torch_load
from stem_plugin.runtime import StemRuntime
from stem_plugin.stem_tasks import TaskConfig
from stem_plugin.toolbox_dataset import ToolboxDataset
from test_tasks import fixture_manifest


def make_runtime(particles=True, semantic=False):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    runtime = StemRuntime(TaskConfig(particles, semantic), backbone='resnet18')
    checkpoint = os.getenv('CEDIRNET_STEM_TEST_LOCALIZATION')
    if particles and checkpoint:
        runtime.load_center(safe_torch_load(checkpoint, map_location='cpu'))
    return runtime


def make_batch(tmp_path, runtime):
    dataset = ToolboxDataset(fixture_manifest(tmp_path, runtime.tasks.nanoparticles),
                             runtime.tasks, size=(64, 64))
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_unchanged(module, before):
    after = module.state_dict()
    assert before.keys() == after.keys()
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert not changed, f'frozen localizer tensors changed: {changed}'


@pytest.mark.parametrize('particles,semantic', [(True, False), (False, True), (True, True)])
def test_real_optimizer_step_preserves_frozen_localizer(tmp_path, particles, semantic):
    runtime = make_runtime(particles, semantic)
    batch = make_batch(tmp_path, runtime)
    optimizer = torch.optim.Adam(runtime.parameters(), lr=1e-4)
    localizer = runtime.center_model.module.instance_center_estimator if particles else None
    frozen = snapshot(localizer) if particles else None
    for _ in range(2):
        runtime.train()
        active = [module for module in (runtime.particle_model, runtime.semantic_model) if module is not None]
        before = [next(module.parameters()).detach().clone() for module in active]
        tracked = [{name: value.clone() for name, value in module.named_buffers()
                    if name.endswith('num_batches_tracked')} for module in active]
        optimizer.zero_grad(set_to_none=True)
        total, parts = runtime.loss(copy.deepcopy(batch))
        assert torch.isfinite(total)
        assert set(parts) == ({'particle_loss'} if particles else set()) | ({'semantic_loss'} if semantic else set())
        torch.testing.assert_close(total, sum(parts.values()), rtol=0, atol=0)
        total.backward()
        optimizer.step()
        for module, old, counters in zip(active, before, tracked):
            assert module.training
            assert not torch.equal(old, next(module.parameters()).detach())
            grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
            assert any(gradient.abs().sum() > 0 for gradient in grads)
            assert counters
            now = dict(module.named_buffers())
            assert all(now[name] > counter for name, counter in counters.items())
        if particles:
            assert not any(parameter.requires_grad or parameter.grad is not None
                           for parameter in runtime.center_model.parameters())
            assert_unchanged(localizer, frozen)
            # Decoding belongs to the outer estimator, not the frozen localizer.
            assert runtime.center_model.training and runtime.center_model.module.training
            assert all(not module.training for module in localizer.modules())
        else:
            assert runtime.center_model is None
        # Trainer.visualize()/the following train_epoch() use this roundtrip.
        runtime.eval()


@pytest.mark.parametrize('semantic', [False, True])
def test_training_loss_preserves_raw_outputs_and_groundtruth(tmp_path, semantic):
    runtime = make_runtime(semantic=semantic).train()
    batch = make_batch(tmp_path, runtime)
    captured = {}

    def capture_raw(module, args, output):
        captured['raw'] = output
        captured['raw_values'] = output.detach().clone()
        output.retain_grad()

    def capture_targets(module, args, output):
        captured['sample'] = copy.deepcopy(output)

    def capture_loss_inputs(module, args, kwargs):
        captured['prediction'] = args[0]
        captured['actual_sample'] = args[1]
        captured['kwargs'] = kwargs

    def capture_losses(module, args, output):
        captured['losses'] = output

    hooks = [runtime.particle_model.register_forward_hook(capture_raw),
             runtime.groundtruth.register_forward_hook(capture_targets),
             runtime.particle_criterion.register_forward_pre_hook(capture_loss_inputs, with_kwargs=True),
             runtime.particle_criterion.register_forward_hook(capture_losses)]
    try:
        total, parts = runtime.loss(batch)
    finally:
        for hook in hooks:
            hook.remove()
    assert captured['prediction'] is captured['raw']
    torch.testing.assert_close(captured['prediction'], captured['raw_values'], rtol=0, atol=0)
    expected_targets = captured['sample']['centerdir_groundtruth']
    actual_targets = captured['actual_sample']['centerdir_groundtruth']
    assert expected_targets.keys() == actual_targets.keys()
    for key, value in expected_targets.items():
        torch.testing.assert_close(actual_targets[key], value, rtol=0, atol=0)
    # The unchanged criterion must see exactly the raw prediction/target domain.
    kwargs = dict(captured['kwargs'], centerdir_gt=expected_targets)
    baseline = runtime.particle_criterion(captured['raw_values'], captured['sample'], **kwargs)
    assert len(baseline) == len(captured['losses'])
    for actual, expected in zip(captured['losses'], baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    torch.testing.assert_close(parts['particle_loss'], baseline[0].mean(), rtol=0, atol=0)
    total.backward()
    gradient = captured['raw'].grad
    assert torch.isfinite(gradient).all()
    assert gradient[:, 2].abs().sum() > 0  # raw log-distance supervision
    assert gradient[:, 3].abs().sum() > 0  # raw log-radius supervision


def direction_fields():
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    dy, dx = 32 - y.float(), 32 - x.float()
    distance = torch.sqrt(dx.square() + dy.square()).clamp_min(1)
    target = torch.stack((dy / distance, dx / distance, torch.zeros_like(dx),
                          torch.ones_like(dx), torch.zeros_like(dx))).unsqueeze(0)
    other = target.clone()
    other[:, :2] = torch.randn((1, 2, 64, 64), generator=torch.Generator().manual_seed(19)) * 3
    return target, other


@pytest.mark.parametrize('mode', ['train', 'eval', 'train-eval-train'])
@torch.no_grad()
def test_localization_is_batch_invariant_in_every_runtime_mode(mode):
    runtime = make_runtime()
    for part in mode.split('-'):
        runtime.train(part == 'train')
    target, other = direction_fields()
    alone = runtime.center_model(target.clone())
    mixed = runtime.center_model(torch.cat((target, other)))
    torch.testing.assert_close(alone['center_heatmap'][0], mixed['center_heatmap'][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alone['center_pred'][0], mixed['center_pred'][0], rtol=1e-5, atol=1e-6)
    # Ensure the regression actually exercises nonzero responses, not empty maps.
    assert alone['center_heatmap'].abs().max() > 0


@torch.no_grad()
def test_serving_checkpoint_keeps_statistics_and_batch_invariant_validation(tmp_path):
    from stem_plugin.serving import load_runtime
    runtime = make_runtime()
    frozen = snapshot(runtime.center_model.module.instance_center_estimator)
    runtime.train().eval().train()
    with torch.enable_grad():
        optimizer = torch.optim.Adam(runtime.parameters(), lr=1e-4)
        loss, _ = runtime.loss(make_batch(tmp_path, runtime))
        loss.backward()
        optimizer.step()
    runtime.eval()
    checkpoint = tmp_path / 'trained.pth'
    torch.save(runtime.checkpoint(0), checkpoint)
    served = load_runtime(dict(model=str(checkpoint), nanoparticles=True, segmentation=False), 'cpu')
    assert not served.training
    assert_unchanged(served.center_model.module.instance_center_estimator, frozen)
    generator = torch.Generator().manual_seed(3)
    images = torch.rand((2, 3, 64, 64), generator=generator) * 255
    # The actual Trainer.visualize and serving paths both call runtime(image) in eval.
    for model in (runtime, served):
        alone = model(images[:1])['particles']
        mixed = model(images)['particles']
        for key in ('output', 'center_heatmap', 'center_pred'):
            torch.testing.assert_close(alone[key][0], mixed[key][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(alone['pred_attributes']['shape_coef'][0],
                                   mixed['pred_attributes']['shape_coef'][0], rtol=1e-5, atol=1e-6)
        arrays = [image.numpy().transpose(1, 2, 0).astype(np.uint8) for image in images]
        single = model.predict(arrays[:1], size=(64, 64), score_threshold=.01)
        multiple = model.predict(arrays, size=(64, 64), score_threshold=.01)
        for key in ('centers', 'scores', 'radii'):
            np.testing.assert_allclose(single[key][0], multiple[key][0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('statistic', ['running_mean', 'running_var'])
def test_checkpoint_missing_normalization_statistics_is_rejected(statistic):
    runtime = make_runtime()
    state = snapshot(runtime.center_model)
    key = next(name for name in state if name.endswith(statistic))
    del state[key]
    with pytest.raises(RuntimeError, match='Missing key'):
        runtime.load_center({'center_model_state_dict': state})
