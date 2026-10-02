"""Training-boundary modality dropout, with real CPU loss/backward/update."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from test_tasks import fixture_manifest

ROOT = Path(__file__).resolve().parents[1]
MODES = [(True, False), (False, True), (True, True)]


def real_trainer(tmp_path, monkeypatch, particles=True, semantic=True, trainer_type=None, **options):
    from train import Trainer
    from stem_plugin.runtime import StemRuntime
    from stem_plugin.toolbox_dataset import ToolboxDataset

    torch.set_num_threads(2)
    trainer = (trainer_type or Trainer)(dict(nanoparticles=particles, segmentation=semantic,
        epochs=2, device='cpu', **options))
    trainer.runtime = StemRuntime(trainer.tasks, device='cpu', backbone='resnet18', pretrained=False)
    manifest = fixture_manifest(tmp_path, particles)
    data = json.loads(manifest.read_text())
    data['train'] *= 4
    manifest.write_text(json.dumps(data))
    dataset = ToolboxDataset(manifest, trainer.tasks, size=(64, 64), augment=True)
    # These are the final augmented batches. A nonzero auxiliary plane makes
    # accidental zeroing/reconstruction visible even though the usual plane is 0.
    trainer.loaders = {'train': list(torch.utils.data.DataLoader(dataset, batch_size=2))}
    for sample in trainer.loaders['train']:
        sample['image'][:, 2].fill_(7)
    trainer.optimizer = torch.optim.Adam(trainer.runtime.parameters(), lr=1e-4)
    monkeypatch.setattr('train.mlflow.log_metrics', lambda *args, **kwargs: None)
    return trainer


@pytest.mark.parametrize('particles,semantic', MODES)
@pytest.mark.parametrize('bf,haadf,dropped', [(1., 0., 0), (0., 1., 1)])
def test_forced_dropout_reaches_all_enabled_models_after_augmentation(
        tmp_path, monkeypatch, particles, semantic, bf, haadf, dropped):
    import train
    from stem_modality import apply_modality_dropout

    trainer = real_trainer(tmp_path, monkeypatch, particles, semantic,
        bf_drop_probability=bf, haadf_drop_probability=haadf)
    batches = trainer.loaders['train']
    originals = [sample['image'].clone() for sample in batches]
    expected = [image.clone() for image in originals]
    for image in expected:
        image[:, dropped].zero_()
    helper_calls = []

    def apply(image, *args, **kwargs):
        helper_calls.append((image, kwargs.get('generator')))
        return apply_modality_dropout(image, *args, **kwargs)

    monkeypatch.setattr(train, 'apply_modality_dropout', apply, raising=False)
    observed = {'particle': [], 'semantic': []}
    handles = []
    for name, model in [('particle', trainer.runtime.particle_model),
                        ('semantic', trainer.runtime.semantic_model)]:
        if model is not None:
            handles.append(model.register_forward_pre_hook(
                lambda module, args, name=name: observed[name].append(args[0].detach().clone())))
    real_loss = trainer.runtime.loss
    received = []

    def loss(sample):
        index = len(received)
        received.append(sample)
        assert torch.equal(sample['image'], expected[index])
        # Labels/metadata must remain the exact final-augmentation objects.
        assert all(sample[key] is value for key, value in batches[index].items() if key != 'image')
        return real_loss(sample)

    monkeypatch.setattr(trainer.runtime, 'loss', loss)
    models = [model for model in (trainer.runtime.particle_model, trainer.runtime.semantic_model)
              if model is not None]
    before = [next(model.parameters()).detach().clone() for model in models]
    metrics = trainer.train_epoch(0)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert all(not torch.equal(old, next(model.parameters()).detach())
               for old, model in zip(before, models))
    assert len(helper_calls) == len(batches)
    assert helper_calls[0][1] is not None and helper_calls[0][1] is helper_calls[1][1]
    for index, sample in enumerate(batches):
        assert torch.equal(sample['image'], originals[index]), 'caller must not be mutated'
    for name, enabled in [('particle', particles), ('semantic', semantic)]:
        assert len(observed[name]) == len(batches) if enabled else not observed[name]
        if enabled:
            assert all(torch.equal(actual, wanted) for actual, wanted in zip(observed[name], expected))
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize('particles,semantic', MODES)
@pytest.mark.parametrize('seed', [None, 41])
def test_defaults_use_private_epoch_stream_once_per_batch(tmp_path, monkeypatch, particles, semantic, seed):
    import train
    from stem_modality import apply_modality_dropout, create_modality_dropout_generator

    trainer = real_trainer(tmp_path, monkeypatch, particles, semantic,
        **({} if seed is None else {'modality_dropout_seed': seed}))
    expected_seed = 0 if seed is None else seed
    calls, generators, received = [], [], []
    originals = [sample['image'].clone() for sample in trainer.loaders['train']]

    def create(seed, epoch):
        before = torch.random.get_rng_state().clone()
        generator = create_modality_dropout_generator(seed, epoch)
        assert torch.equal(before, torch.random.get_rng_state())
        generators.append((seed, epoch, generator))
        return generator

    def apply(image, bf_drop_probability, haadf_drop_probability, generator=None):
        before = torch.random.get_rng_state().clone()
        result = apply_modality_dropout(image, bf_drop_probability, haadf_drop_probability, generator)
        assert torch.equal(before, torch.random.get_rng_state()), 'dropout must not consume global RNG'
        calls.append((bf_drop_probability, haadf_drop_probability, generator))
        return result

    monkeypatch.setattr(train, 'create_modality_dropout_generator', create, raising=False)
    monkeypatch.setattr(train, 'apply_modality_dropout', apply, raising=False)
    real_loss = trainer.runtime.loss

    def loss(sample):
        received.append(sample['image'].clone())
        return real_loss(sample)

    monkeypatch.setattr(trainer.runtime, 'loss', loss)
    for epoch in range(2):
        trainer.train_epoch(epoch)
    assert [(seed, epoch) for seed, epoch, _ in generators] == [(expected_seed, 0), (expected_seed, 1)]
    assert len(calls) == 4
    expected = []
    for epoch in range(2):
        generator = create_modality_dropout_generator(expected_seed, epoch)
        expected.extend(apply_modality_dropout(image, .25, .25, generator) for image in originals)
        epoch_calls = calls[2*epoch:2*epoch+2]
        assert all(bf == .25 and haadf == .25 and stream is generators[epoch][2]
                   for bf, haadf, stream in epoch_calls)
    assert all(torch.equal(actual, wanted) for actual, wanted in zip(received, expected))
    assert any(not torch.equal(actual, original) for actual, original in zip(received[:2], originals))


@pytest.mark.parametrize('enabled', [False, 'false'])
def test_disabled_preserves_final_batch_identity_and_never_calls_helper(tmp_path, monkeypatch, enabled):
    import train
    trainer = real_trainer(tmp_path, monkeypatch, modality_dropout=enabled)

    def forbidden(*args, **kwargs):
        pytest.fail('disabled dropout must not create a generator, copy input, or draw RNG')

    monkeypatch.setattr(train, 'create_modality_dropout_generator', forbidden, raising=False)
    monkeypatch.setattr(train, 'apply_modality_dropout', forbidden, raising=False)
    real_loss = trainer.runtime.loss
    received = []

    def loss(sample):
        original = trainer.loaders['train'][len(received)]
        received.append(sample)
        assert sample is original and sample['image'] is original['image']
        return real_loss(sample)

    monkeypatch.setattr(trainer.runtime, 'loss', loss)
    trainer.train_epoch(0)
    assert len(received) == 2


@pytest.mark.parametrize('options', [
    {'modality_dropout': 'no'}, {'modality_dropout': 1}, {'modality_dropout': 0},
    {'modality_dropout': []}, {'modality_dropout': {}},
    {'bf_drop_probability': float('nan')}, {'haadf_drop_probability': float('inf')},
    {'bf_drop_probability': -.1}, {'haadf_drop_probability': -.1},
    {'bf_drop_probability': .75, 'haadf_drop_probability': .5},
    {'bf_drop_probability': 'bad'},
    {'modality_dropout': False, 'bf_drop_probability': .9, 'haadf_drop_probability': .9},
])
def test_invalid_options_fail_before_initialization(options):
    from train import Trainer
    with pytest.raises(ValueError, match='true or false|probabilities'):
        Trainer(options)


@pytest.mark.parametrize('particles,semantic', MODES)
@pytest.mark.parametrize('enabled', [True, False])
def test_checkpoint_records_upstream_policy_without_changing_tensors(
        tmp_path, monkeypatch, particles, semantic, enabled):
    import train
    from stem_modality import modality_dropout_metadata

    trainer = real_trainer(tmp_path, monkeypatch, particles, semantic,
        modality_dropout=enabled, bf_drop_probability=.1, haadf_drop_probability=.3,
        modality_dropout_seed=42)
    before = trainer.runtime.checkpoint(1)
    before = {key: {name: tensor.clone() for name, tensor in value.items()}
              if key.endswith('state_dict') else value for key, value in before.items()}
    saved = []
    monkeypatch.delenv('MLFLOW_ARTIFACTS_DESTINATION', raising=False)
    monkeypatch.setattr(train.torch, 'save', lambda state, path: saved.append(state))
    monkeypatch.setattr(train.mlflow, 'active_run', lambda: types.SimpleNamespace(
        info=types.SimpleNamespace(experiment_id='1', run_id='test-run')))
    monkeypatch.setattr(train.mlflow, 'log_artifact', lambda *args, **kwargs: None)
    monkeypatch.setattr('modelargs.emit_action', lambda *args, **kwargs: None)
    trainer.checkpoint(1)
    state = saved[0]
    assert state['modality_dropout'] == dict(modality_dropout_metadata(.1, .3, 42), enabled=enabled)
    assert state.keys() == before.keys() | {'modality_dropout'}
    for key, value in before.items():
        if key.endswith('state_dict'):
            assert state[key].keys() == value.keys()
            assert all(torch.equal(state[key][name], tensor) for name, tensor in value.items())
        else:
            assert state[key] == value
    clone = type(trainer.runtime)(trainer.tasks, device='cpu', backbone='resnet18', pretrained=False)
    clone.load(state, inference=True)


@pytest.mark.parametrize('options,exception,message', [
    (['--bf_drop_probability', 'nan'], ValueError, 'probabilities'),
    (['--haadf_drop_probability', 'inf'], ValueError, 'probabilities'),
    (['--bf_drop_probability', '-.1'], ValueError, 'probabilities'),
    (['--bf_drop_probability', '.8', '--haadf_drop_probability', '.3'], ValueError, 'probabilities'),
    (['--modality_dropout', 'false', '--bf_drop_probability', '.8', '--haadf_drop_probability', '.3'],
     ValueError, 'probabilities'),
    (['--modality_dropout', 'invalid'], SystemExit, None),
])
def test_invalid_cli_options_do_not_initialize_or_create_run_artifacts(
        tmp_path, monkeypatch, options, exception, message):
    import train
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--manifest', str(tmp_path / 'missing.json'), *options])
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'sqlite:///' + str(tmp_path / 'runs.db'))
    monkeypatch.setenv('MLFLOW_ARTIFACTS_DESTINATION', str(tmp_path / 'artifacts'))

    def forbidden(*args, **kwargs):
        pytest.fail('invalid options must fail before initialization or run setup')

    monkeypatch.setattr(train.Trainer, 'initialize', forbidden)
    monkeypatch.setattr(train.mlflow, 'set_tracking_uri', forbidden)
    monkeypatch.setattr(train.mlflow, 'start_run', forbidden)
    with pytest.raises(exception, match=message):
        train.main()
    assert not (tmp_path / 'runs.db').exists()
    assert not (tmp_path / 'artifacts').exists()


@pytest.mark.parametrize('subset', ['training', 'validation'])
def test_visualization_never_ablates_paired_inputs(tmp_path, monkeypatch, subset):
    import train
    trainer = real_trainer(tmp_path, monkeypatch, particles=False, semantic=True,
        bf_drop_probability=1., haadf_drop_probability=0.)
    from stem_plugin.toolbox_dataset import ToolboxDataset
    dataset = ToolboxDataset(tmp_path / 'manifest.json', trainer.tasks, size=(64, 64))
    trainer.loaders[subset] = torch.utils.data.DataLoader(dataset, batch_size=2)

    def forbidden(*args, **kwargs):
        pytest.fail('dropout must be train-epoch only')

    monkeypatch.setattr(train, 'apply_modality_dropout', forbidden)
    monkeypatch.setattr(train, 'create_modality_dropout_generator', forbidden)
    monkeypatch.setattr(train, 'log_figure_artifact', lambda *args, **kwargs: None)
    received = []
    handle = trainer.runtime.semantic_model.register_forward_pre_hook(
        lambda module, args: received.append(args[0].clone()))
    trainer.visualize(0, subset)
    handle.remove()
    assert len(received) == 2
    assert all(torch.all(image[:, 0] == 17) and torch.all(image[:, 1] == 91) for image in received)


def test_schema_exposes_train_only_default_enabled_options():
    properties = json.loads((ROOT / 'model.json').read_text())['properties']
    expected = {'modality_dropout': ('boolean', True),
        'bf_drop_probability': ('number', .25), 'haadf_drop_probability': ('number', .25),
        'modality_dropout_seed': ('integer', 0)}
    for key, (kind, default) in expected.items():
        assert key in properties, f'missing train option {key}'
        assert properties[key]['stage'] == 'train'
        assert properties[key]['type'] == kind
        assert properties[key]['default'] == default
    for key in ['bf_drop_probability', 'haadf_drop_probability']:
        assert properties[key]['minimum'] == 0 and properties[key]['maximum'] == 1


def test_setup_verifies_public_helper_import():
    setup = (ROOT / 'setup.sh').read_text()
    verification = setup.split('PYTHONPATH=', 1)[1]
    assert 'from stem_modality import' in verification
    assert 'apply_modality_dropout' in verification
    assert 'create_modality_dropout_generator' in verification
    assert 'modality_dropout_metadata' in verification
    assert 'validate_probabilities' in verification
    assert 'branch=master' in setup
    assert not (ROOT / 'stem_modality.py').exists(), 'use upstream, do not vendor helper'


def test_exact_training_cli_launch_completes_with_default_policy(tmp_path):
    manifest = fixture_manifest(tmp_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONPATH='.',
        MLFLOW_TRACKING_URI='sqlite:///' + str(tmp_path / 'runs.db'),
        MLFLOW_ARTIFACTS_DESTINATION=str(tmp_path / 'artifacts'), MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, 'train.py', '--manifest', str(manifest),
        '--nanoparticles', 'false', '--segmentation', 'true', '--width', '64', '--height', '64',
        '--batch_size', '1', '--workers', '0', '--epochs', '1', '--save_interval', '1',
        '--display_interval', '1', '--visualization_samples', '1'], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1/1:' in result.stderr and '100%' in result.stderr
    assert 'Toolbox:Weights:' in result.stdout
    checkpoints = list((tmp_path / 'artifacts').rglob('checkpoint.pth'))
    assert len(checkpoints) == 1
    state = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
    from stem_modality import modality_dropout_metadata
    assert state['modality_dropout'] == dict(modality_dropout_metadata(.25, .25, 0), enabled=True)
    assert state['tasks']['segmentation'] and not state['tasks']['nanoparticles']
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=env['MLFLOW_TRACKING_URI'])
    experiment = client.get_experiment_by_name('CeDiRNet-STEM')
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == 'FINISHED'
    assert runs[0].data.params['manifest'] == str(manifest)
    assert 'manfest' not in runs[0].data.params


@pytest.mark.parametrize('particles,semantic', MODES)
def test_disabled_matches_untouched_base_trainer_exactly(tmp_path, monkeypatch, particles, semantic):
    source = subprocess.check_output(['git', 'show', '02aa62d:cedirnet-stem/train.py'],
        cwd=ROOT, text=True)
    baseline = types.ModuleType('baseline_stem_train')
    exec(compile(source, '<baseline-train.py>', 'exec'), baseline.__dict__)
    snapshots = []
    for trainer_type, options in [(baseline.Trainer, {}), (None, {'modality_dropout': False})]:
        torch.manual_seed(125)
        trainer = real_trainer(tmp_path, monkeypatch, particles, semantic,
            trainer_type=trainer_type, **options)
        # Exercise the actual random loader augmentation and model RNG, not a
        # same-implementation omitted-vs-false comparison.
        from stem_plugin.toolbox_dataset import ToolboxDataset
        dataset = ToolboxDataset(tmp_path / 'manifest.json', trainer.tasks, size=(64, 64), augment=True)
        trainer.loaders['train'] = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        inputs = []
        real_loss = trainer.runtime.loss

        def loss(sample):
            inputs.append(sample['image'].clone())
            return real_loss(sample)

        trainer.runtime.loss = loss
        metrics = trainer.train_epoch(0)
        state = {name: tensor.clone() for name, tensor in trainer.runtime.state_dict().items()}
        snapshots.append((inputs, metrics, state, torch.random.get_rng_state().clone()))
        del trainer
    old_inputs, old_metrics, old_state, old_rng = snapshots[0]
    new_inputs, new_metrics, new_state, new_rng = snapshots[1]
    assert len(old_inputs) == len(new_inputs)
    assert all(torch.equal(old, new) for old, new in zip(old_inputs, new_inputs))
    assert old_metrics == new_metrics
    assert old_state.keys() == new_state.keys()
    assert all(torch.equal(old_state[name], new_state[name]) for name in old_state)
    assert torch.equal(old_rng, new_rng)
