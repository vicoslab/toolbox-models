"""Run with plugin and upstream src on PYTHONPATH."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


def fixture_manifest(tmp_path, points=False):
    for name, value in [('bf', 17), ('haadf', 91), ('mask', 1)]:
        Image.new('L', (64, 64), value).save(tmp_path / f'{name}.png')
    item = dict(images=['bf.png', 'haadf.png'], semantic_mask='mask.png')
    if points:
        item['points'] = [[20, 24, 4]]
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(version=2, semantic_classes=['Carbon','Film','Vacuum'], train=[item], val=[item])))
    return path


def test_flag_parser_and_neither():
    from task_options import task_config
    assert not task_config({'nanoparticles':'false', 'segmentation':'true'}).nanoparticles
    with pytest.raises(ValueError, match='at least one'):
        task_config({'nanoparticles':'false','segmentation':'false'})
    with pytest.raises(ValueError, match='true or false'):
        task_config({'nanoparticles':'no'})


@pytest.mark.parametrize('particles,semantic', [(True,False),(False,True),(True,True)])
def test_manifest_modes(tmp_path, particles, semantic):
    from stem_tasks import TaskConfig
    from toolbox_dataset import ToolboxDataset
    ds = ToolboxDataset(fixture_manifest(tmp_path, particles), TaskConfig(particles, semantic), size=(32,32))
    sample = ds[0]
    assert sample['image'].shape == (3,32,32)
    assert sample['image'][0,0,0] == 17
    assert ('semantic_segmentation' in sample) == semantic
    assert ('center' in sample) == particles
    if semantic:
        assert sample['semantic_segmentation'].unique().tolist() == [1]


def test_missing_supervision_and_class_ids(tmp_path):
    from stem_tasks import TaskConfig
    from toolbox_dataset import ToolboxDataset
    manifest = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match='points'):
        ToolboxDataset(manifest, TaskConfig(True,False))
    Image.new('L',(64,64),7).save(tmp_path/'mask.png')
    with pytest.raises(ValueError, match='class id'):
        ToolboxDataset(manifest, TaskConfig(False,True))[0]


@pytest.mark.parametrize('particles,semantic', [(True,False),(False,True),(True,True)])
def test_runtime_train_checkpoint_infer(tmp_path, particles, semantic):
    import torch
    from stem_tasks import TaskConfig
    from runtime import StemRuntime
    from toolbox_dataset import ToolboxDataset
    torch.set_num_threads(2)
    tasks = TaskConfig(particles,semantic)
    dataset = ToolboxDataset(fixture_manifest(tmp_path,particles),tasks,size=(64,64))
    sample = next(iter(torch.utils.data.DataLoader(dataset,batch_size=1)))
    runtime = StemRuntime(tasks,device='cpu',backbone='resnet18',pretrained=False)
    optimizer = torch.optim.Adam(runtime.parameters(),lr=1e-4)
    before = next(runtime.parameters()).detach().clone()
    runtime.train()
    loss,parts = runtime.loss(sample)
    assert torch.isfinite(loss)
    loss.backward(); optimizer.step()
    assert not torch.equal(before,next(runtime.parameters()).detach())
    state = runtime.checkpoint(0)
    assert ('semantic_model_state_dict' in state) == semantic
    assert ('center_model_state_dict' in state) == particles
    clone = StemRuntime(tasks,device='cpu',backbone='resnet18',pretrained=False)
    clone.load(state, inference=True)
    clone.eval()
    with torch.no_grad():
        output = clone.predict([np.zeros((40,80,3),dtype=np.uint8)],size=(64,64),score_threshold=.5)
    assert len(output['centers']) == 1
    assert len(output['segmentation']) == 1
    assert (output['segmentation'][0] is not None) == semantic
    if not particles:
        assert output['centers'] == [[]]
    wrong = dict(state); wrong['tasks'] = dict(state['tasks'],classes=['x','y','z'])
    if semantic:
        with pytest.raises(ValueError,match='classes'):
            clone.load(wrong,inference=True)


@pytest.mark.parametrize('particles,semantic', [(True,False),(False,True),(True,True)])
def test_trainer_real_epoch(tmp_path, monkeypatch, particles, semantic):
    import torch
    import mlflow
    from train import Trainer
    torch.set_num_threads(2)
    monkeypatch.setenv('MLFLOW_ARTIFACTS_DESTINATION',str(tmp_path/'artifacts'))
    mlflow.set_tracking_uri('sqlite:///'+str(tmp_path/'runs.db'))
    mlflow.set_experiment('stem-smoke')
    options = dict(manifest=str(fixture_manifest(tmp_path,particles)),nanoparticles=str(particles).lower(),segmentation=str(semantic).lower(),
        width=64,height=64,batch_size=1,workers=0,epochs=1,save_interval=1,
        display_interval=1,visualization_samples=1,backbone='resnet18',device='cpu')
    if particles:
        from runtime import StemRuntime
        from stem_tasks import TaskConfig
        initial=StemRuntime(TaskConfig(particles,semantic),backbone='resnet18')
        path=tmp_path/'initial-fixture.pth'; torch.save(initial.checkpoint(-1),path)
        options['model']=str(path)
        del initial
    with mlflow.start_run():
        trainer = Trainer(options)
        trainer.initialize(); trainer.run()
    assert list((tmp_path/'artifacts').rglob('checkpoint.pth'))
    assert len(list((tmp_path/'artifacts').rglob('*diagnostics.png'))) == 2


@pytest.mark.parametrize('particles,semantic', [(True,False),(False,True),(True,True)])
def test_http_inference_and_preannotations(tmp_path, monkeypatch, particles, semantic):
    import io
    import sys
    import importlib.util
    import torch
    from runtime import StemRuntime
    from stem_tasks import TaskConfig
    from serving import preannotation
    torch.set_num_threads(2)
    tasks=TaskConfig(particles,semantic)
    runtime=StemRuntime(tasks,backbone='resnet18')
    path=tmp_path/'weights.pth'; torch.save(runtime.checkpoint(0),path)
    root=Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys,'argv',['infer','--model',str(path),'--nanoparticles',str(particles).lower(),
        '--segmentation',str(semantic).lower(),'--width','64','--height','64'])
    monkeypatch.setenv('MODEL_DIR',str(tmp_path/'backend-cache'))
    import label_studio_ml.api
    importlib.reload(label_studio_ml.api)  # backend keeps one global Flask app per process
    spec=importlib.util.spec_from_file_location('http_infer',root/'infer.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    client=module.app.test_client()
    assert client.post('/infer').status_code == 400
    def image(value):
        buf=io.BytesIO(); Image.new('L',(80,40),value).save(buf,format='PNG'); buf.seek(0); return buf
    result=client.post('/infer',data={'images':[(image(17),'BF.png'),(image(91),'HAADF.png')]})
    assert result.status_code == 200, result.json
    assert result.json['tasks'] == tasks.to_dict()
    config={'labels':dict(type='Labels',to_name=['image'],labels=['Particle']),
            'semantic':dict(type='BrushLabels',to_name=['image'],labels=list(tasks.classes))}
    annotation=preannotation(result.json,0,(80,40),config)
    assert all(x['type']=='brushlabels' for x in annotation['result']) if not particles else True
    if semantic:
        assert any(x['type']=='brushlabels' for x in annotation['result'])


def test_particle_metrics_match_radius_and_empty():
    from validation_metrics import ParticleMetrics
    metric = ParticleMetrics(distance=3)
    metric.update([[10,10],[50,50]],[5,2],[[11,10]],[4])
    values=metric.compute()
    assert values['particle_precision']==.5
    assert values['particle_recall']==1
    assert values['particle_radius_mae_px']==1
    assert values['particle_localization_mae_px']==1
    assert ParticleMetrics().compute()['particle_f1']==0


def test_semantic_output_roundtrip():
    from semantic_results import encode_mask, brush_results
    from label_studio_converter.brush import decode_rle
    mask = np.array([[0,1],[2,2]],dtype=np.uint8)
    result = encode_mask(mask, ['Carbon','Film','Vacuum'])
    assert result['width'] == 2 and result['classes'][2] == 'Vacuum'
    brushes = brush_results(mask, ['Carbon','Film','Vacuum'], 'semantic', 'image')
    assert len(brushes) == 3
    assert np.array(decode_rle(brushes[2]['value']['rle'])).reshape(2,2,4)[:,:,3].tolist() == [[0,0],[255,255]]


def test_diagnostics_modes(tmp_path):
    from diagnostics import plot_training_diagnostics, training_artifact_path
    from matplotlib import pyplot as plt
    for particles, semantic in [(True,False),(False,True),(True,True)]:
        fig = plot_training_diagnostics(image=np.zeros((3,32,32)), centers=[[10,10]], radii=[3], scores=[.9],
            direction_output=np.ones((5,32,32)) if particles else None,
            localization_response=np.zeros((32,32)) if particles else None,
            semantic_target=np.ones((32,32)) if semantic else None,
            semantic_prediction=np.ones((32,32)) if semantic else None,
            classes=['Carbon','Film','Vacuum'])
        fig.savefig(tmp_path/f'{particles}-{semantic}.png'); plt.close(fig)
    assert '..' not in training_artifact_path(0,'../../a.png','training')
