"""Exercise stock STEM groundtruth, weighted ShapeLoss and actual gradients.

Run with upstream STEM's src on PYTHONPATH; an installed plugin environment
already supplies it. No target/loss/network stand-ins are used in these tests.
"""
import copy
import json
import os

from PIL import Image
import pytest
import torch

from stem_plugin.runtime import StemRuntime
from stem_plugin.stem_tasks import TaskConfig
from stem_plugin.toolbox_dataset import ToolboxDataset


@pytest.fixture(scope='module')
def runtime():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    model = StemRuntime(TaskConfig(True, False), backbone='resnet18')
    checkpoint = os.environ.get('CEDIRNET_STEM_CENTER_CHECKPOINT')
    if checkpoint:
        model.load_center(torch.load(checkpoint, map_location='cpu', weights_only=False))
    return model


def sample(tmp_path, positives, size=(64, 64)):
    for name, value in [('BF.png', 17), ('HAADF.png', 91)]:
        Image.new('L', (64, 64), value).save(tmp_path / name)
    items = [dict(images=['BF.png', 'HAADF.png'],
                  points=[[32, 32, 8]] if positive else []) for positive in positives]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 4, 'train': items}))
    dataset = ToolboxDataset(manifest, TaskConfig(True, False), size=size)
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(items))))


def groundtruth(runtime, sample):
    return runtime.groundtruth(sample, torch.arange(len(sample['image']), dtype=torch.int32))


def loss_kwargs(runtime, sample):
    return dict(centerdir_gt=sample['centerdir_groundtruth'],
                ignore_mask=sample['ignore'] > 0,
                difficult_mask=torch.zeros_like(sample['instance'].squeeze(1)),
                reduction_dims=(1, 2, 3), **runtime.loss_w)


def stock_criterion(runtime):
    from criterions import get_criterion
    from stem_plugin.base_config import get_args
    args = get_args()
    return get_criterion(args['loss_type'], args['loss_opts'],
                         runtime.particle_model.module, runtime.center_model.module)


@pytest.mark.parametrize('positives', [(False,), (False, False), (False, False, False, False), (False, True), (True, False), (True, True)])
def test_distance_only_mask_with_stock_loss_and_gradients(tmp_path, runtime, positives):
    batch = groundtruth(runtime, sample(tmp_path, positives))
    predicted = torch.full((len(positives), 5, 64, 64), .2, requires_grad=True)
    kwargs = loss_kwargs(runtime, batch)
    baseline_criterion = stock_criterion(runtime)
    baseline = baseline_criterion(predicted, batch, **kwargs)
    # The unwrapped upstream criterion must succeed, before any loss adjustment.
    assert all(torch.isfinite(value).all() for value in baseline)
    assert baseline[0].shape == (len(positives),)
    adjusted = runtime.particle_criterion(predicted, batch, **kwargs)
    baseline_grad, = torch.autograd.grad(baseline[0].sum(), predicted, retain_graph=True)
    adjusted_grad, = torch.autograd.grad(adjusted[0].sum(), predicted)
    assert torch.isfinite(baseline_grad).all()
    assert torch.isfinite(adjusted_grad).all()
    negative = ~torch.tensor(positives, dtype=torch.bool)
    expected_direction = baseline[4] + baseline[5] + baseline[7]
    expected_total = baseline[1] + expected_direction + baseline[3] + baseline[8]
    adjusted_values = {0: expected_total, 2: expected_direction, 6: torch.zeros_like(baseline[6])}
    for index, (old, new) in enumerate(zip(baseline, adjusted)):
        expected = torch.where(negative, adjusted_values[index], old) if index in adjusted_values else old
        torch.testing.assert_close(new, expected, rtol=0, atol=0)
    for b, positive in enumerate(positives):
        if positive:
            torch.testing.assert_close(adjusted_grad[b], baseline_grad[b], rtol=0, atol=0)
            assert adjusted_grad[b, 2].abs().sum() > 0
            assert adjusted_grad[b, 3].abs().sum() > 0
            assert not adjusted_grad[b, 3][batch['label'][b, 0] == 0].any()
        else:
            assert baseline_grad[b, 2].abs().sum() > 0
            assert adjusted_grad[b, 2].abs().sum() == 0
            assert adjusted_grad[b, 3:].abs().sum() == 0
            torch.testing.assert_close(adjusted_grad[b, :2], baseline_grad[b, :2], rtol=0, atol=0)
            assert (adjusted_grad[b, :2].abs().sum((1, 2)) > 0).all()
            for key in ('gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_shape_coef'):
                assert not batch['centerdir_groundtruth'][key][b].any()
    diagnostics = runtime.particle_criterion.get_loss_dict(adjusted)
    torch.testing.assert_close(diagnostics['loss'], adjusted[0].sum())
    torch.testing.assert_close(diagnostics['losses_tasks']['centerdir'], adjusted[2].sum())
    torch.testing.assert_close(diagnostics['losses_main']['r'], adjusted[6].sum())


@pytest.mark.parametrize('all_ignore', [False, True])
def test_raw_criterion_handles_rectangular_all_negative_batches(tmp_path, runtime, all_ignore):
    batch = sample(tmp_path, (False, False, False, False), size=(96, 64))
    if all_ignore:
        batch['ignore'].fill_(1)
    batch = groundtruth(runtime, batch)
    height, width = batch['label'].shape[-2:]
    assert height != width
    predicted = torch.full((4, 5, height, width), .2, requires_grad=True)
    raw = stock_criterion(runtime)(predicted, batch, **loss_kwargs(runtime, batch))
    assert raw[0].shape == (4,)
    assert all(torch.isfinite(v).all() for v in raw)
    raw_grad, = torch.autograd.grad(raw[0].sum(), predicted, retain_graph=True)
    assert torch.isfinite(raw_grad).all()
    losses = runtime.particle_criterion(predicted, batch, **loss_kwargs(runtime, batch))
    losses[0].sum().backward()
    assert torch.isfinite(predicted.grad).all()
    assert not predicted.grad[:, 2:].any()
    if all_ignore:
        assert not predicted.grad.any()
        assert not losses[0].any()
    else:
        assert (predicted.grad[:, :2] > 0).all()


def test_negative_shape_loss_is_defined_directly_in_runtime():
    import stem_plugin.runtime as module
    wrapper = getattr(module, 'NegativeImageShapeLoss', None)
    assert wrapper is not None, 'runtime.py must define NegativeImageShapeLoss'
    assert wrapper.__module__ == module.__name__
    from pathlib import Path
    assert not (Path(module.__file__).parent / 'particle_loss.py').exists()


def test_positive_groundtruth_is_identical_to_stock(tmp_path, runtime):
    from models.center_groundtruth import CenterDirGroundtruth
    from stem_plugin.base_config import get_args
    batch = sample(tmp_path, (True, True))
    opts = get_args()['train_dataset']['centerdir_gt_opts']
    # A groundtruth generator caches its coordinate grid across resolutions.
    # Compare fresh stock/adapted instances, not one with prior rectangular crops.
    stock = CenterDirGroundtruth(**opts)
    adapted = type(runtime.groundtruth)(**opts)
    indexes = torch.arange(2, dtype=torch.int32)
    baseline = stock(copy.deepcopy(batch), indexes)['centerdir_groundtruth']
    actual = adapted(batch, indexes)['centerdir_groundtruth']
    assert actual.keys() == baseline.keys()
    for key in baseline:
        torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)


def test_large_undefined_distance_cannot_cancel_negative_direction_loss(tmp_path, runtime):
    batch = groundtruth(runtime, sample(tmp_path, (False, True)))
    predicted = torch.full((2, 5, 64, 64), .2, requires_grad=True)
    with torch.no_grad():
        predicted[0, 2] = 1e10
    losses = runtime.particle_criterion(predicted, batch, **loss_kwargs(runtime, batch))
    expected_direction = losses[4][0] + losses[5][0] + losses[7][0]
    assert expected_direction > 0
    torch.testing.assert_close(losses[2][0], expected_direction, rtol=0, atol=0)
    torch.testing.assert_close(losses[0][0], expected_direction + losses[1][0]
                               + losses[3][0] + losses[8][0], rtol=0, atol=0)
    losses[0].sum().backward()
    assert predicted.grad[0, :2].abs().sum() > 0
    assert not predicted.grad[0, 2:].any()


@pytest.mark.parametrize('all_ignore', [False, True])
def test_ignored_negative_pixels_have_no_gradient(tmp_path, runtime, all_ignore):
    batch = sample(tmp_path, (False, True))
    batch['ignore'][0, :, :8, :] = 1
    if all_ignore:
        batch['ignore'][0] = 1
    batch = groundtruth(runtime, batch)
    predicted = torch.full((2, 5, 64, 64), .2, requires_grad=True)
    losses = runtime.particle_criterion(predicted, batch, **loss_kwargs(runtime, batch))
    losses[0].mean().backward()
    assert torch.isfinite(predicted.grad).all()
    assert not predicted.grad[0, :, :8, :].any()
    assert not predicted.grad[0, 2:].any()
    if all_ignore:
        assert losses[0][0] == 0
        assert not predicted.grad[0].any()
    else:
        assert predicted.grad[0, :2, 8:, :].abs().sum() > 0
    assert predicted.grad[1, 3].abs().sum() > 0


def test_distance_validity_comes_from_generated_targets_in_both_plugins(tmp_path, runtime):
    batch = groundtruth(runtime, sample(tmp_path, (False, True)))
    # Neither support rasters nor padded coordinates may override the S/C
    # supervision already generated for the criterion.
    batch['instance'].zero_()
    batch['center'][0, 5] = torch.tensor([40., 40.])
    batch['center'][1] = 0
    predicted = torch.full((2, 5, 64, 64), .2, requires_grad=True)
    baseline = stock_criterion(runtime)(predicted, batch, **loss_kwargs(runtime, batch))
    actual = runtime.particle_criterion(predicted, batch, **loss_kwargs(runtime, batch))
    assert actual[6][0] == 0
    torch.testing.assert_close(actual[6][1], baseline[6][1], rtol=0, atol=0)
    assert actual[6][1] > 0


@pytest.mark.parametrize('positives', [(False,), (False, False), (False, False, False, False), (False, True), (True, False), (True, True)])
def test_real_runtime_forward_backward_preserves_mean(tmp_path, runtime, positives):
    batch = sample(tmp_path, positives)
    runtime.zero_grad(set_to_none=True)
    outputs, losses_seen = [], []
    def retain_output(module, inputs, output):
        output.retain_grad()
        outputs.append(output)
    def retain_losses(module, inputs, output):
        losses_seen.append(output)
    model_hook = runtime.particle_model.register_forward_hook(retain_output)
    loss_hook = runtime.particle_criterion.register_forward_hook(retain_losses)
    try:
        total, parts = runtime.loss(batch)
        torch.testing.assert_close(total, losses_seen[0][0].mean(), rtol=0, atol=0)
        torch.testing.assert_close(parts['particle_loss'], total, rtol=0, atol=0)
        assert torch.isfinite(total)
        total.backward()
    finally:
        model_hook.remove()
        loss_hook.remove()
    gradient = outputs[0].grad
    assert torch.isfinite(gradient).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in runtime.particle_model.parameters())
    assert not any(p.grad is not None for p in runtime.center_model.parameters())
    for b, positive in enumerate(positives):
        assert (gradient[b, :2].abs().sum((1, 2)) > 0).all()
        if positive:
            assert gradient[b, 2].abs().sum() > 0
            assert gradient[b, 3].abs().sum() > 0
        else:
            assert not gradient[b, 2:].any()


def test_joint_negative_keeps_all_ignore_semantic_loss_zero(tmp_path):
    torch.set_num_threads(2)
    runtime = StemRuntime(TaskConfig(True, True), backbone='resnet18')
    batch = sample(tmp_path, (False,))
    batch['semantic_segmentation'] = torch.full((1, 64, 64), 255, dtype=torch.long)
    total, parts = runtime.loss(batch)
    assert torch.isfinite(total)
    assert parts['semantic_loss'] == 0
    assert parts['particle_loss'] > 0
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in runtime.particle_model.parameters())
    semantic_gradients = [p.grad for p in runtime.semantic_model.parameters() if p.grad is not None]
    assert semantic_gradients
    assert all(torch.isfinite(g).all() and not g.any() for g in semantic_gradients)
