import os
import site
site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/cedirnet')
site.addsitedir('src')

import collections
import json
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from criterions import get_criterion
from datasets import get_centerdir_dataset
from models import get_model, get_center_model, CenterOrientationEstimator
from utils import transforms as my_transforms
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval
from utils.hard_sampler import HardExamplesBatchSampler
from utils.utils import AverageMeter, variable_len_collate

from models.multitask_model import MultiTaskModel
from criterions.loss_weighting.weight_methods import get_weight_method

import modelargs
import mlflow
from extras import plot_results, load_center_model

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if self.args['cuda'] else 'cpu')

    def _to_data_parallel(self, X, **kwargs):
        return torch.nn.DataParallel(X.to(self.device), device_ids=[0], **kwargs)

    def initialize(self):
        args = self.args

        ###################################################################################################
        # train dataloader
        dataset_workers = args['train_dataset'].get('workers', 0)
        self.dataset_batch = args['train_dataset'].get('batch_size', 1)
        dataset_shuffle = args['train_dataset'].get('shuffle', True)
        dataset_hard_sample_size = args['train_dataset'].get('hard_samples_size')

        self.accumulate_grads_iter = args['model'].get('accumulate_grads_iter',1)
        if self.accumulate_grads_iter:
            self.dataset_batch = self.dataset_batch // self.accumulate_grads_iter

        train_dataset, self.centerdir_groundtruth_op = get_centerdir_dataset('', args['train_dataset']['kwargs'], args['train_dataset'].get('centerdir_gt_opts'))

        # prepare hard-examples sampler for dataset
        if dataset_shuffle:
            default_sampler = torch.utils.data.RandomSampler(train_dataset)
        else:
            default_sampler = torch.utils.data.SequentialSampler(train_dataset)

        self.batch_sampler = HardExamplesBatchSampler(train_dataset,
                                                    default_sampler,
                                                    batch_size=self.dataset_batch,
                                                    hard_sample_size=dataset_hard_sample_size,
                                                    drop_last=True,
                                                    hard_samples_selected_min_percent=args['train_dataset'].get('hard_samples_selected_min_percent'),
                                                    hard_samples_only_min_selected_when_empty=args['train_dataset'].get('hard_samples_only_min_selected_when_empty'),
                                                    device=self.device)

        self.train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_sampler=self.batch_sampler,
                                                    num_workers=dataset_workers, pin_memory=True if args['cuda'] else False,
                                                    collate_fn=variable_len_collate)

        self.model = get_model(args['model']['name'], args['model']['kwargs'])
        self.model.init_output(args['loss_opts']['num_vector_fields'])

        self.center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'], is_learnable=args['center_model'].get('use_learnable_center_estimation', True))
        # so we can use it as center estimator with orientation even though it isn't
        self.center_model.enable_6dof = args.get('enable_6dof')
        self.center_model.use_orientation_confidence_score = args.get('use_orientation_confidence_score')

        self.center_model.init_output(args['loss_opts']['num_vector_fields'])

        self.criterion = get_criterion(args.get('loss_type'), args.get('loss_opts'), self.model, self.center_model)

        self.multitask_weighting = None
        mw = args.get('multitask_weighting')
        if mw and mw['name'] != 'off':

            assert isinstance(model, MultiTaskModel)
            self.multitask_weighting = get_weight_method(mw['name'], device=self.device, **mw['kwargs'])

        if self.centerdir_groundtruth_op is not None:
            self.centerdir_groundtruth_op = self._to_data_parallel(self.centerdir_groundtruth_op)

        self.model = self._to_data_parallel(self.model, dim=0)
        self.center_model = self._to_data_parallel(self.center_model, dim=0)
        self.criterion = self._to_data_parallel(self.criterion, dim=0)

        def get_optimizer(model_, args_):
            if args_ is None or args_.get('disabled'):
                return None, None
            if 'optimizer' not in args_ or args_['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model_.parameters(),lr=args_['lr'],
                                             weight_decay=args_['weight_decay'])
            elif args_['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(model_.parameters(),lr=args_['lr'],
                                            momentum=args_['momentum'],
                                            weight_decay=args_['weight_decay'])
            # use custom lambda_scheduler_fn function that can pass args if available
            lr_lambda = args_['lambda_scheduler_fn'](args) if 'lambda_scheduler_fn' in args_ else args_['lambda_scheduler']
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            return optimizer, scheduler

        # set optimizer for model and for center model
        self.optimizer, self.scheduler = get_optimizer(self.model, args['model'])
        self.center_optimizer, self.center_scheduler = get_optimizer(self.center_model, args['center_model'])

        self.sample_loss_history = {}
        self.sample_centerdir_loss_history = {}
        self.sample_hard_neg_selection = {}

        # resume
        self.start_epoch = 0
        resume_path = args.get('resume_path')
        if resume_path and os.path.exists(resume_path):
            print('Resuming model from {}'.format(resume_path))
            state = torch.load(resume_path)
            self.start_epoch = state['epoch'] + 1

            if model_dict := state.get('model_state_dict'):
                self.model.load_state_dict(model_dict, strict=True)
            
            if optim_dict := state.get('optim_state_dict') and self.optimizer:
                self.optimizer.load_state_dict(optim_dict)
            
            if criterion_dict := state.get('criterion_state_dict'):
                self.criterion.load_state_dict(criterion_dict)
            
            if center_model_dict := state.get('center_model_state_dict') and args['center_model'].get('use_learnable_center_estimation'):
                self.center_model.load_state_dict(center_model_dict, strict=True)
            
            if center_optim_dict := state.get('center_optim_state_dict') and self.center_optimizer:
                self.center_optimizer.load_state_dict(center_optim_dict)

        pretrained_model_path = args.get('pretrained_model_path')
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print('Loading pre-trained model from {}'.format(pretrained_model_path))
            state = torch.load(pretrained_model_path)

            if 'model_state_dict' in state:
                INPUT_WEIGHTS_KEY = 'module.model.encoder.model.stem_0.weight'
                if checkpoint_input_weights := state['model_state_dict'].get(INPUT_WEIGHTS_KEY):
                    model_input_weights = self.model.module.model.encoder.model.stem_0.weight
                    if checkpoint_input_weights.shape[1] < model_input_weights.shape[1]:
                        weights = torch.zeros_like(model_input_weights)
                        weights[:, :checkpoint_input_weights.shape[1], :, :] = checkpoint_input_weights
                        state['model_state_dict'][INPUT_WEIGHTS_KEY] = weights

                        print('WARNING: #####################################################################################################')
                        print(f'WARNING: pretrained model input shape mismatch - will load weights for only the first {checkpoint_input_weights.shape[1]} channels, is this correct ?!!!')
                        print('WARNING: #####################################################################################################')

                missing, unexpected = model.load_state_dict(state['model_state_dict'], strict=False)
                if len(missing) > 0 or len(unexpected) > 0:
                    print('WARNING: #####################################################################################################')
                    print('WARNING: Current model differs from the pretrained one, loading weights using strict=False')
                    print('WARNING: #####################################################################################################')

            if 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                self.center_model.load_state_dict(state['center_model_state_dict'], strict=True)

        
        if center_model_path := args.get('pretrained_center_model_path'):
            print('Loading pre-trained center model from {}'.format(center_model_path))
            state = torch.load(center_model_path)

            INPUT_WEIGHTS_KEY = 'module.instance_center_estimator.conv_start.0.weight'
            if (checkpoint_input_weights := state['center_model_state_dict'].get(INPUT_WEIGHTS_KEY)) is not None:
                center_input_weights = self.center_model.module.instance_center_estimator.conv_start[0].weight
                if checkpoint_input_weights.shape != center_input_weights.shape:
                    state['center_model_state_dict'][INPUT_WEIGHTS_KEY] = checkpoint_input_weights[:, :2, :, :]

                    print('WARNING: #####################################################################################################')
                    print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                    print('WARNING: #####################################################################################################')

            self.center_model.load_state_dict(state['center_model_state_dict'], strict=False)

        self.denormalize_args = None

        # get prepare values/functions needed for display
        if transforms := args['train_dataset']['kwargs'].get('transform'):
            if isinstance(transforms,my_transforms.Compose):
                for t in transforms.transforms:
                    if type(t) == my_transforms.Normalize and 'image' in t.keys:
                        self.denormalize_args = (t.mean[t.keys == 'image'], t.std[t.keys == 'image'])
                        break
            elif isinstance(transforms,list):
                for t in transforms:
                    opts = t['opts']
                    if t['name'] == 'Normalize' and 'image' in opts['keys']:
                        self.denormalize_args = (opts['mean'][opts['keys'] == 'image'], opts['std'][opts['keys'] == 'image'])
                        break

        self.log_r_fn = self.criterion.module.log_r_fn if getattr(self.criterion.module, 'use_log_r', False) else None
        if args['loss_opts'].get('learnable_center_loss', '') == 'cross-entropy':
            self.center_conv_resp_fn = lambda x: torch.sigmoid(x)
        else:
            self.center_conv_resp_fn = lambda x: x #torch.relu(x)

    def train(self, epoch):
        n_epochs = args['n_epochs']

        # put model into training mode
        self.model.train()
        self.center_model.train()

        iter=epoch*len(self.train_dataset_it)

        all_samples_metrics = {}
        tqdm_iterator = tqdm(self.train_dataset_it, desc=f'{epoch}/{n_epochs}',dynamic_ncols=True)

        for i, sample in enumerate(tqdm_iterator):

            # call centerdir_groundtruth_op first which will create any missing centerdir_groundtruth (using GPU) and add synthetic output
            if self.centerdir_groundtruth_op is not None:
                sample = self.centerdir_groundtruth_op(sample, torch.arange(0, self.dataset_batch).int())

            im = sample['image']

            instances = sample['instance'].squeeze(dim=1)
            ignore = sample.get('ignore')
            centerdir_gt = sample.get('centerdir_groundtruth')

            from models.center_groundtruth import CenterDirGroundtruth

            loss_ignore = None
            if ignore is not None:
                # treat any type of ignore objects (truncated, border, etc) as ignore during training
                # (i.e., ignore loss and any groundtruth objects at those pixels)
                loss_ignore = ignore > 0

            # get difficult mask based on ignore flags (VALUE of 8 == difficult flag and VALUE of 2 == truncated flag )
            difficult = (((ignore & 8) | (ignore & 2)) > 0).squeeze(dim=1) if ignore is not None else torch.zeros_like(instances)

            # get gt_centers from centerdir_gt and convert them to dictionary (filter-out non visible and ignored examples)
            gt_centers = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_centers'])
            gt_centers_dict = CenterDirGroundtruth.convert_gt_centers_to_dictionary(gt_centers, instances=instances, ignore=loss_ignore)

            # retrieve and set random seed for hard examples from previous epoch
            # (will be returned as None if sample does not exist or is not hard-sample)
            sample['seed'] = self.batch_sampler.retrieve_hard_sample_storage_batch(sample['index'],'seed')

            # call center prediction model
            center_output = self.center_model(self.model(im), **sample)
            output, center_pred, center_heatmap = [center_output[k] for k in ['output','center_pred','center_heatmap']]

            # get losses
            losses = self.criterion(output, sample,
                            centerdir_responses=(center_pred, center_heatmap), centerdir_gt=centerdir_gt, ignore_mask=loss_ignore,
                            difficult_mask=difficult, reduction_dims=(1,2,3), epoch_percent=epoch/n_epochs, **self.args['loss_w'])

            sample_metrics = None
            if self.batch_sampler.has_hard_samples():
                # evaluate predictions to get metrics needed for difficulty score
                sample_metrics = self._evaluate_batch_predictions(center_pred, gt_centers_dict, ignore, difficult)

                # calc difficulty score from losses and metrics
                sample_difficulty_score = self._calc_sample_difficulty(losses, sample_metrics)

                # pass losses to hard-samples batch sampler
                self.batch_sampler.update_difficulty_score(sample, sample_difficulty_score, index_key='index', storage_keys=['seed'])

            # save losses and metrics from this batch to common storage for this epoch
            all_samples_metrics = self._updated_per_epoch_sample_metrics(all_samples_metrics, sample['index'],
                                                                         losses, sample_metrics)
            loss = losses[0].sum()

            # we can simply sum the final loss since average is already calculated through weighting
            if self.multitask_weighting is None:
                bp_loss = loss / self.accumulate_grads_iter
                bp_loss.backward()
            else:
                losses_tasks = self.criterion.module.get_loss_dict(losses).get('losses_tasks')
                assert losses_tasks, 'Criterion is not compatible with multitask weighting (missing "task_losses" key in returned get_loss_dict())'

                losses_tasks = [t_loss.sum() / self.accumulate_grads_iter for t_loss in losses_tasks.values()]

                bp_loss, extra_outputs = self.multitask_weighting(
                    losses=torch.stack(losses_tasks),
                    shared_parameters=list(self.model.module.shared_parameters()),
                    task_specific_parameters=list(self.model.module.task_specific_parameters()),
                    last_shared_parameters=list(self.model.module.last_shared_parameters()),
                    representation=None, # TODO: add representation if needed at all
                )

            if ((i + 1) % self.accumulate_grads_iter == 0) or (i + 1 == len(self.train_dataset_it)):
                if self.optimizer:
                    self.optimizer.step()
                    self.optimizer.zero_grad() # set_to_none=False for prior to v2.0 and set_to_none=True after v2.0

                if self.center_optimizer:
                    self.center_optimizer.step()
                    self.center_optimizer.zero_grad()

            metrics = OrderedDict(loss=loss.item())
            if tqdm_iterator is not None:
                loss_dict = self.criterion.module.get_loss_dict(losses)
                metrics.update({n: l.cpu().item() for n,l in loss_dict['losses_tasks' if 'losses_tasks' in loss_dict else 'losses_groups'].items()})

                tqdm_iterator.set_postfix(**metrics)
            
            mlflow.log_metrics(metrics, epoch)

            iter+=1

        all_samples_total_loss = {k:v['loss'] for k,v in all_samples_metrics.items()}

        return np.array(list(all_samples_total_loss.values())).mean() * self.dataset_batch

    def _calc_sample_difficulty(self, losses, metrics):
        loss_total = losses[0]

        sample_difficulty_score = torch.zeros((len(losses[0]),), dtype=torch.float, device=losses[0].device)

        for b in range(len(loss_total)):
            FP, FN = metrics[b]['FP'], metrics[b]['FN']

            # multiply loss with number of FP and FN for hard neg
            sample_difficulty_score[b] = loss_total[b].sum() * (FP + 2 * FN + 1) ** 2
            # sample_difficulty_score[b] = loss_total[b].sum() * (FN) ** 2

        return sample_difficulty_score

    def _evaluate_batch_predictions(self, center_pred, gt_centers_dict, ignore, difficult):
        metrics = [{} for _ in range(len(center_pred))]

        for b in range(len(center_pred)):
            # calc FP and FN
            if center_pred is not None:
                center_eval = CenterGlobalMinimizationEval()
                valid_pred = center_pred[b, center_pred[b, :, 0] != 0, :]
                valid_pred = valid_pred[ignore[b, 0, valid_pred[:, 2].long(), valid_pred[:, 1].long()] == 0, :] if ignore is not None else valid_pred
                center_eval.add_image_prediction(None, None, None, valid_pred[:, 1:3].cpu().numpy(), None, None,
                                                 gt_centers_dict[b], difficult[b], None)
                FP, FN = center_eval.metrics['FP'][0], center_eval.metrics['FN'][0]
            else:
                FP, FN = 0, 0

            metrics[b] = dict(FP=FP, FN=FN)

        return metrics

    def _updated_per_epoch_sample_metrics(self, stored_results, sample_indexes, losses, metrics=None):
        loss_total, loss_cls, loss_centerdir_total, loss_centers, loss_sin, loss_cos, loss_r, loss_magnitude_reg = losses[:8]

        for b in range(len(loss_total)):
            index = sample_indexes[b].item()

            # add losses
            stored_results[index] = dict(loss=loss_total[b].sum().item(),
                                         loss_centerdir=loss_centerdir_total[b].sum().item() + loss_cls[b].sum().item())

            if metrics is not None:
                stored_results[index].update(metrics[b])

        return stored_results

    def run(self):
        args = self.args

        for epoch in range(self.start_epoch, args['n_epochs']):

            train_loss = self.train(epoch)

            if self.scheduler: self.scheduler.step()
            if self.center_scheduler: self.center_scheduler.step()

            if args['display'] and epoch % args['display_it'] == 0:
                with torch.no_grad():
                    for sample in tqdm(self.train_dataset_it, desc='visualise', dynamic_ncols=True):

                        center_output = self.center_model(self.model(sample['image']), **sample)
                        direction_maps, center_pred, center_heatmap, angle_pred = map(lambda k: center_output[k].detach().cpu().numpy(), ['output', 'center_pred', 'center_heatmap', 'pred_angle'])
                        
                        for name, im, centers, angles, dirs in zip(sample['name'], sample['image'], center_pred, angle_pred, direction_maps):
                            valid = centers[:, 0] == 1
                            scores = centers[valid, -1]

                            fig, _ = plot_results(im.cpu().numpy().transpose((1,2,0)), centers[valid, 1:-1], scores, angles[valid])
                            mlflow.log_figure(fig, name)
                            plt.close(fig)

            if args['save'] and (epoch % args.get('save_interval',10) == 0 or epoch + 1 == args['n_epochs']):
                print('Saving checkpoint', flush=True)
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model and self.model.state_dict(),
                    # 'optim_state_dict': self.optimizer and self.optimizer.state_dict(),
                    # 'criterion_state_dict': self.criterion and self.criterion.state_dict(),
                    'center_model_state_dict': self.center_model and self.center_model.state_dict(),
                    # 'center_optim_state_dict': self.center_optimizer and self.center_optimizer.state_dict(),
                }

                with tempfile.TemporaryDirectory() as d:
                    filename = os.path.join(d, "checkpoint.pth")
                    torch.save(state, filename)
                    mlflow.log_artifact(filename)

if __name__ == '__main__':
    from base_config import args

    cmd_args = modelargs.parse('./model.json')

    args['train_dataset']['kwargs']['manifest'] = cmd_args['manifest']
    args['n_epochs'] = cmd_args['epochs']
    
    mlflow.set_tracking_uri('http://localhost:8081')
    mlflow.set_experiment('CeDiRNet')
    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run(run_name=cmd_args.get('name')) as run:
        print(f'Experiment {run.info.experiment_id}: Run {run.info.run_id}') # this gets parsed and turned into a link in the frontend
        mlflow.log_params(json.loads(json.dumps(args, default=lambda _: '<not serializable>')))

        trainer = Trainer(args)

        trainer.initialize()
        trainer.run()
