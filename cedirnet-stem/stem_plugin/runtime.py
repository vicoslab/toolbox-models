"""Shared optional-task runtime for training, inference and diagnostics."""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .stem_tasks import TaskConfig, build_semantic_model
from .checkpoint import load_compatible_model_state
from .results import restore_prediction
from .semantic_results import encode_mask


class StemRuntime(torch.nn.Module):
    def __init__(self, tasks, device='cpu', backbone='tu-convnext_base', pretrained=False):
        super().__init__()
        self.tasks, self.device, self.backbone = tasks, torch.device(device), backbone
        self.particle_model = self.center_model = self.semantic_model = None
        if tasks.nanoparticles:
            from .base_config import get_args, NUM_VECTOR_FIELDS
            from models import get_model, get_center_model
            from models.center_groundtruth import CenterDirGroundtruth
            from criterions import get_criterion
            args = get_args()
            args['model']['kwargs'].update(backbone=backbone,pretrained=pretrained)
            if backbone == 'resnet18':
                args['model']['kwargs']['fpn_args']['encoder_depth'] = 5
            model = get_model(args['model']['name'],args['model']['kwargs'])
            model.init_output(NUM_VECTOR_FIELDS)
            self.particle_model = torch.nn.DataParallel(model.to(self.device))
            center = get_center_model(args['center_model']['name'],args['center_model']['kwargs'],is_learnable=True)
            center.init_output(NUM_VECTOR_FIELDS)
            self.center_model = torch.nn.DataParallel(center.to(self.device))
            # Stock localization weights are fixed, as in upstream STEM training.
            self.center_model.requires_grad_(False)
            class ParticleGroundtruth(CenterDirGroundtruth):
                # Stock groundtruth omits regression maps for negative images.
                # Supply shaped zeros before its variable-length batch collation,
                # for both all-negative and mixed positive/negative batches.
                def _generate_center_directions(self, sample, b, *values):
                    maps = super()._generate_center_directions(sample, b, *values)
                    for key in ('gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th'):
                        if key not in maps:
                            maps[key] = torch.zeros_like(sample['label'][b], dtype=torch.float)
                    return maps

                def _generate_custom_regression_map(self, sample, b, *values):
                    maps = super()._generate_custom_regression_map(sample, b, *values)
                    if 'gt_shape_coef' not in maps:
                        maps['gt_shape_coef'] = torch.zeros_like(sample['shape_coef'][b]).unsqueeze(1)
                    return maps

            self.groundtruth = ParticleGroundtruth(**args['train_dataset']['centerdir_gt_opts']).to(self.device)
            self.particle_criterion = get_criterion(args['loss_type'],args['loss_opts'],model,center).to(self.device)
            self.loss_w = args['loss_w']
        if tasks.segmentation:
            from .semantic_loss import MulticlassCrossEntropyDiceLoss
            self.semantic_model = build_semantic_model(tasks,backbone=backbone,pretrained=pretrained,
                encoder_depth=5 if backbone=='resnet18' else 4).to(self.device)
            self.semantic_criterion = MulticlassCrossEntropyDiceLoss([1.0]*len(tasks.classes)).to(self.device)

    def train(self, mode=True):
        # The estimator's training path preserves log-radius regression targets;
        # eval mutates them for display. Frozen parameters do not imply eval here.
        super().train(mode)
        return self

    def forward(self, image):
        image = image.to(self.device)
        out = {}
        if self.particle_model is not None:
            out['particles'] = self.center_model(self.particle_model(image),detect_centers=True)
        if self.semantic_model is not None:
            out['semantic'] = self.semantic_model(image)
        return out

    def loss(self, sample):
        sample = {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in sample.items()}
        parts = {}
        if self.tasks.nanoparticles:
            sample = self.groundtruth(sample,torch.arange(sample['image'].shape[0],dtype=torch.int32,device=self.device))
            out = self.center_model(self.particle_model(sample['image']),**sample)
            losses = self.particle_criterion(out['output'],sample,
                centerdir_responses=(out['center_pred'],out['center_heatmap']),
                centerdir_gt=sample['centerdir_groundtruth'],ignore_mask=sample['ignore']>0,
                difficult_mask=torch.zeros_like(sample['instance'].squeeze(1)),
                reduction_dims=(1,2,3),**self.loss_w)
            parts['particle_loss'] = losses[0].mean()
        if self.tasks.segmentation:
            losses = self.semantic_criterion(self.semantic_model(sample['image']),sample)
            parts['semantic_loss'] = losses[0].mean()
        return sum(parts.values()),parts

    def checkpoint(self, epoch):
        state = dict(format_version=2,epoch=epoch,tasks=self.tasks.to_dict(),backbone=self.backbone)
        if self.tasks.nanoparticles:
            state.update(model_state_dict=self.particle_model.state_dict(),center_model_state_dict=self.center_model.state_dict())
        if self.tasks.segmentation:
            state['semantic_model_state_dict'] = self.semantic_model.state_dict()
        return state

    def load_center(self, state):
        values = state.get('center_model_state_dict')
        if not values:
            raise ValueError('checkpoint does not contain center_model_state_dict')
        values = dict(values)
        key = 'module.instance_center_estimator.conv_start.0.weight'
        expected = self.center_model.state_dict()
        # The released 3DoF localizer consumes C/S plus magnitude and class
        # channels. STEM disables the latter two; no other shape change is safe.
        if (key in values and tuple(values[key].shape) == (16, 4, 3, 3)
                and tuple(expected[key].shape) == (16, 2, 3, 3)):
            values[key] = values[key][:, :2]

        # Old checkpoints serialized coordinate caches and the analytic 1D
        # kernels. This runtime has no instance-mask estimator/augmentation and
        # uses the learned 2D localizer, not those 1D kernels. Never filter by
        # prefix or discard arbitrary unmatched learned weights.
        legacy_geometry = {
            'module.instance_mask_estimator.xym_1024',
            'module.center_augmentator.xym',
            'module.instance_center_estimator.kernel_cos',
            'module.instance_center_estimator.kernel_sin',
        }
        ignored = sorted((values.keys() - expected.keys()) & legacy_geometry)
        for name in ignored:
            del values[name]
        if ignored:
            logging.getLogger(__name__).warning('Ignoring legacy localization geometry: %s', ', '.join(ignored))

        # Like main's strict=False loader, retain runtime-initialized smoothing
        # when absent from the legacy checkpoint; no kernel regeneration needed.
        # Allow only this known nonlearned tensor, not missing learned weights.
        gaussian_key = 'module.instance_center_estimator.gaussian_blur.conv.weight'
        if gaussian_key in expected and gaussian_key not in values:
            values[gaussian_key] = expected[gaussian_key]
        self.center_model.load_state_dict(values,strict=True)

    def load(self, state, inference=False):
        saved = TaskConfig.from_dict(state['tasks']) if 'tasks' in state else None
        if saved is not None and state.get('backbone',self.backbone) != self.backbone:
            raise ValueError('checkpoint backbone mismatch')
        if self.tasks.nanoparticles:
            if saved is not None and not saved.nanoparticles:
                raise ValueError('checkpoint has no nanoparticle task')
            if saved is None:
                load_compatible_model_state(self.particle_model,state)
            else:
                self.particle_model.load_state_dict(state['model_state_dict'],strict=True)
            if state.get('center_model_state_dict'):
                self.load_center(state)
            elif inference:
                raise ValueError('particle inference requires localization weights')
        if self.tasks.segmentation:
            semantic = state.get('semantic_model_state_dict')
            # Also accept exact semantic research checkpoints, not particle logits.
            metadata = state.get('metadata',{})
            if semantic is None and metadata.get('class_names') == list(self.tasks.classes) and metadata.get('ignore_index') == self.tasks.ignore_index:
                semantic = state.get('model_state_dict')
            if saved is not None and (not saved.segmentation or saved.classes != self.tasks.classes):
                raise ValueError('checkpoint semantic classes/tasks mismatch')
            if semantic is None:
                raise ValueError('checkpoint has no semantic weights; train segmentation first')
            self.semantic_model.load_state_dict(semantic,strict=True)

    @torch.no_grad()
    def predict(self, images, size=(512,512), score_threshold=.5):
        if not images:
            return dict(centers=[],scores=[],radii=[],segmentation=[],tasks=self.tasks.to_dict())
        tensors = [torch.from_numpy(np.array(Image.fromarray(im).resize(size,Image.Resampling.BILINEAR)).transpose(2,0,1).copy()).float() for im in images]
        out = self(torch.stack(tensors))
        response = dict(centers=[],scores=[],radii=[],segmentation=[],tasks=self.tasks.to_dict())
        for i,image in enumerate(images):
            h,w = image.shape[:2]
            centers,scores,radii = [],[],[]
            if self.tasks.nanoparticles:
                pred = out['particles']['center_pred'][i].detach().cpu().numpy()
                radius = out['particles']['pred_attributes']['shape_coef'][i].detach().cpu().numpy()
                for j in np.argsort(pred[:,4])[::-1]:
                    if pred[j,0] != 1 or pred[j,4] < score_threshold:
                        continue
                    center,r = restore_prediction(pred[j,1:3],float(radius[j,0]),size,(w,h))
                    if np.isfinite(r) and r > 0:
                        centers.append(list(center)); scores.append(float(pred[j,4])); radii.append(r)
            semantic = None
            if self.tasks.segmentation:
                logits = F.interpolate(out['semantic'][i:i+1],size=(h,w),mode='bilinear',align_corners=False)
                semantic = encode_mask(logits.argmax(1)[0].cpu().numpy(),self.tasks.classes)
            response['centers'].append(centers); response['scores'].append(scores)
            response['radii'].append(radii); response['segmentation'].append(semantic)
        return response
