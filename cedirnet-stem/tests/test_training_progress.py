"""Exercise real training and tqdm output, including non-TTY stderr."""
import json

import pytest
import torch

from test_tasks import fixture_manifest


@pytest.mark.parametrize('particles,semantic', [(True, False), (False, True), (True, True)])
def test_training_progress_on_captured_stderr(tmp_path, monkeypatch, capsys, particles, semantic):
    from train import Trainer
    from stem_plugin.runtime import StemRuntime
    from stem_plugin.toolbox_dataset import ToolboxDataset
    torch.set_num_threads(2)
    manifest = fixture_manifest(tmp_path, particles)
    data = json.loads(manifest.read_text())
    data['train'] *= 2
    manifest.write_text(json.dumps(data))
    trainer = Trainer(dict(nanoparticles=particles, segmentation=semantic, epochs=2))
    trainer.runtime = StemRuntime(trainer.tasks, device='cpu', backbone='resnet18', pretrained=False)
    dataset = ToolboxDataset(manifest, trainer.tasks, size=(64, 64))
    trainer.loaders = {'train': torch.utils.data.DataLoader(dataset, batch_size=1)}
    trainer.optimizer = torch.optim.Adam(trainer.runtime.parameters(), lr=1e-4)
    logged = []
    monkeypatch.setattr('train.mlflow.log_metrics', lambda metrics, step: logged.append((metrics, step)))
    capsys.readouterr()
    before = next(trainer.runtime.parameters()).detach().clone()
    for epoch in range(2):
        metrics = trainer.train_epoch(epoch)
        output = capsys.readouterr()
        assert f'{epoch + 1}/2:' in output.err
        assert '\r' in output.err
        assert '100%' in output.err and '2/2' in output.err
        assert 'loss=' in output.err
        assert ('particle_loss=' in output.err) == particles
        assert ('semantic_loss=' in output.err) == semantic
        assert f'{epoch + 1}/2:' not in output.out
        assert logged[-1] == (metrics, epoch + 1)
        assert metrics['loss'] == pytest.approx(sum(v for k, v in metrics.items() if k != 'loss'))
    assert not torch.equal(before, next(trainer.runtime.parameters()).detach())
