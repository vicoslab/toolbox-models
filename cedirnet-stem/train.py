"""Toolbox training with independently enabled STEM tasks and MLflow views."""
import os
import site
from pathlib import Path
site.addsitedir(str(Path(os.environ.get('CEDIRNET_STEM_SOURCE',os.path.join(os.environ.get('TOOLBOX_CACHE','.'),'cedirnet-stem'))) / 'src'))

import signal
import tempfile
import numpy as np
import torch
import mlflow
from matplotlib import pyplot as plt
from stem_plugin.task_options import task_config
from stem_plugin.toolbox_dataset import ToolboxDataset
from stem_plugin.runtime import StemRuntime
from stem_plugin.checkpoint import safe_torch_load
from stem_plugin.diagnostics import plot_training_diagnostics, training_artifact_path
from stem_plugin.validation_metrics import ParticleMetrics


def log_figure_artifact(fig, artifact_file):
    run = mlflow.active_run()
    artifacts = os.getenv('MLFLOW_ARTIFACTS_DESTINATION')
    if artifacts and run:
        destination = Path(artifacts)/run.info.experiment_id/run.info.run_id/'artifacts'/artifact_file
        destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination)
    else:
        mlflow.log_figure(fig,artifact_file)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.tasks = task_config(args)
        self.device = args.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = (int(args.get('width',512)),int(args.get('height',512)))
        if any(s <= 0 or s % 32 for s in self.size):
            raise ValueError('image dimensions must be positive multiples of 32')
        for key in ('epochs','batch_size','save_interval','display_interval','visualization_samples'):
            if int(args.get(key,1)) < 1:
                raise ValueError(f'{key} must be positive')

    def initialize(self):
        args = self.args
        self.runtime = StemRuntime(self.tasks,self.device,args.get('backbone') or 'tu-convnext_base',pretrained=False)
        if args.get('model'):
            self.runtime.load(safe_torch_load(args['model'],map_location=self.device))
        if self.tasks.nanoparticles:
            path = args.get('localisation')
            if not path and not args.get('model'):
                path = os.path.join(os.environ.get('TOOLBOX_CACHE','.'),'cedirnet-stem','localization_checkpoint.pth')
            if path:
                self.runtime.load_center(safe_torch_load(path,map_location=self.device))
        self.loaders = {}
        for subset,split,augment in [('train','train',True),('training','train',False),('validation','val',False)]:
            dataset = ToolboxDataset(args['manifest'],self.tasks,split,self.size,augment,
                                     allow_empty=subset=='validation')
            self.loaders[subset] = torch.utils.data.DataLoader(dataset,
                batch_size=int(args.get('batch_size',2)),shuffle=augment,
                num_workers=int(args.get('workers') or 0))
        self.optimizer = torch.optim.Adam((p for p in self.runtime.parameters() if p.requires_grad),lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda epoch: max(0,1-epoch/int(args.get('epochs',100)))**.9)

    def train_epoch(self, epoch):
        self.runtime.train()
        values = []
        for sample in self.loaders['train']:
            self.optimizer.zero_grad(set_to_none=True)
            loss,parts = self.runtime.loss(sample)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite STEM training loss')
            loss.backward(); self.optimizer.step()
            values.append({k:float(v.detach()) for k,v in parts.items()})
        metrics = {key:float(np.mean([v[key] for v in values])) for key in values[0]}
        metrics['loss'] = sum(metrics.values())
        mlflow.log_metrics(metrics,step=epoch+1)
        return metrics

    @torch.no_grad()
    def visualize(self, epoch, subset):
        self.runtime.eval()
        limit = int(self.args.get('visualization_samples',4))
        threshold = float(self.args.get('score_threshold',.5))
        confusion = np.zeros((len(self.tasks.classes),len(self.tasks.classes)),dtype=np.int64)
        visualized = 0
        particle_metrics = ParticleMetrics(distance=20)
        for sample in self.loaders[subset]:
            output = self.runtime(sample['image'])
            for i,name in enumerate(sample['name']):
                kwargs = dict(image=sample['image'][i],classes=self.tasks.classes)
                if self.tasks.nanoparticles:
                    out = output['particles']
                    pred = out['center_pred'][i].cpu().numpy()
                    valid = (pred[:,0]==1)&(pred[:,4]>=threshold)
                    kwargs.update(centers=pred[valid,1:3],scores=pred[valid,4],
                        radii=out['pred_attributes']['shape_coef'][i].cpu().numpy()[valid],
                        direction_output=out['output'][i],localization_response=out['center_heatmap'][i],
                        ground_truth_centers=sample['center'][i])
                    gt = sample['center'][i].numpy()
                    gt = gt[np.any(gt != 0, axis=1)]
                    gt_radii = [sample['shape_coef'][i,0,int(y),int(x)].item() for x,y in gt]
                    particle_metrics.update(kwargs['centers'],kwargs['radii'],gt,gt_radii)
                if self.tasks.segmentation:
                    pred = output['semantic'][i].argmax(0).cpu().numpy()
                    target = sample['semantic_segmentation'][i].numpy()
                    valid = target != 255
                    n = len(self.tasks.classes)
                    confusion += np.bincount(n*target[valid]+pred[valid],minlength=n*n).reshape(n,n)
                    kwargs.update(semantic_target=target,semantic_prediction=pred)
                if visualized < limit:
                    fig = plot_training_diagnostics(**kwargs)
                    try:
                        log_figure_artifact(fig,training_artifact_path(epoch,f'{visualized:04d}-{name}',subset))
                    finally:
                        plt.close(fig)
                    visualized += 1
        if self.tasks.nanoparticles and len(self.loaders[subset].dataset):
            mlflow.log_metrics({f'{subset}/{key}':value for key,value in particle_metrics.compute().items()},step=epoch+1)
        if self.tasks.segmentation and confusion.sum():
            union = confusion.sum(0)+confusion.sum(1)-confusion.diagonal()
            present = union>0
            iou = np.divide(confusion.diagonal(),union,out=np.zeros(len(union),float),where=present)
            metrics = {f'{subset}/semantic_mIoU':float(iou[present].mean()),
                       f'{subset}/semantic_pixel_accuracy':float(confusion.trace()/confusion.sum())}
            metrics.update({f'{subset}/semantic_iou_class_{i}':float(iou[i]) for i in np.flatnonzero(present)})
            mlflow.log_metrics(metrics,step=epoch+1)

    def checkpoint(self, epoch):
        state = self.runtime.checkpoint(epoch)
        run = mlflow.active_run()
        artifacts = os.getenv('MLFLOW_ARTIFACTS_DESTINATION')
        relative = 'checkpoints/checkpoint.pth'
        if artifacts:
            filename = Path(artifacts)/run.info.experiment_id/run.info.run_id/'artifacts'/relative
            filename.parent.mkdir(parents=True,exist_ok=True)
            torch.save(state,filename)
        else:
            with tempfile.TemporaryDirectory() as directory:
                filename = Path(directory)/'checkpoint.pth'
                torch.save(state,filename); mlflow.log_artifact(str(filename),artifact_path='checkpoints')
        import modelargs
        modelargs.emit_action('Weights',f'mlflow-artifacts:/{run.info.experiment_id}/{run.info.run_id}/artifacts/{relative}')

    def run(self):
        epochs = int(self.args.get('epochs',100))
        for epoch in range(epochs):
            self.train_epoch(epoch); self.scheduler.step()
            if (epoch+1)%int(self.args.get('display_interval',10))==0 or epoch+1==epochs:
                self.visualize(epoch,'training'); self.visualize(epoch,'validation')
            if (epoch+1)%int(self.args.get('save_interval',10))==0 or epoch+1==epochs:
                self.checkpoint(epoch)


def main():
    import modelargs
    args = modelargs.parse('./model.json')
    trainer = Trainer(args)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','http://localhost:8081'))
    mlflow.set_experiment('CeDiRNet-STEM')
    with mlflow.start_run(run_name=args.get('name')) as run:
        def handler(_signal,_frame):
            mlflow.end_run('KILLED')
            raise SystemExit(0)
        signal.signal(signal.SIGINT,handler); signal.signal(signal.SIGTERM,handler)
        modelargs.emit_action('Experiment',run.info.experiment_id)
        modelargs.emit_action('Run',run.info.run_id)
        mlflow.log_params({k:v for k,v in args.items() if v is not None})
        trainer.initialize(); trainer.run()


if __name__ == '__main__':
    main()
