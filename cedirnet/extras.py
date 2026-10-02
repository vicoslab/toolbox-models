import copy

from matplotlib import pyplot as plt
from matplotlib.patheffects import SimpleLineShadow, Normal
from models import get_center_model
import numpy as np
import torch

def plot_results(image, centers, scores, angles, dist=30):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    fig.tight_layout()
    show_3dof = False  # Reserved for a future explicit inference visualization option.
    for index, ((x, y, _), score) in enumerate(zip(centers, scores)):
        if show_3dof and angles is not None:
            angle = np.deg2rad(angles[index])
            dx, dy = np.cos(angle)*dist, np.sin(angle)*dist
            ax.annotate('', xytext=(x, y), xy=(x+dx, y+dy), arrowprops=dict(color='lime', arrowstyle='->'))
            ax.annotate(f'{score:.2f}', xy=(x-dx/4, y-dy/4), size='xx-small', ha='center', va='center', c='lime', path_effects=[
                SimpleLineShadow(shadow_color="black", linewidth=1, offset=(0,0), alpha=0.7),
                Normal()
            ])
        else:
            ax.scatter(x, y, s=11.25, c='#1677ff', edgecolors='white', linewidths=1.5, alpha=0.75, zorder=3)
    return fig, ax

def center_model_kwargs(args):
    """Keep checkpoint BN buffers, independently of the training policy."""
    kwargs = copy.deepcopy(args['center_model']['kwargs'])
    kwargs.setdefault('dilated_nn_args', {})['freeze_learning'] = False
    return kwargs


def set_center_model_mode(center_model, *, training, freeze_learning=False):
    """Use batch statistics only for the original frozen-localizer train path.

    Retain the pretrained buffers without updating them during that path.
    Eval always uses those buffers. Keep the parent estimator in train mode
    for losses: its eval path decodes regression outputs/targets in place.
    """
    center_model.train(training)
    model = center_model
    while hasattr(model, 'module'):
        model = model.module
    for module in model.instance_center_estimator.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.running_mean is None or module.running_var is None:
                raise ValueError('localizer requires checkpoint running statistics; reconstruct with tracking enabled')
            module.track_running_stats = not (training and freeze_learning)


def load_center_state(center_model, state):
    """Load learned weights and real BN statistics; allow known legacy geometry."""
    values = dict(state['center_model_state_dict'])
    expected = center_model.state_dict()
    first = 'module.instance_center_estimator.conv_start.0.weight'
    if (first in values and tuple(values[first].shape) == (16, 4, 3, 3)
            and tuple(expected[first].shape) == (16, 2, 3, 3)):
        values[first] = values[first][:, :2]

    statistics = [key for key in expected if key.endswith(('.running_mean', '.running_var'))]
    for key in statistics:
        value = values.get(key)
        if (value is None or not torch.isfinite(value).all()
                or (key.endswith('.running_var') and (value < 0).any())):
            raise ValueError(
                f'localization checkpoint requires valid BatchNorm running statistics ({key}); '
                'use the original localization checkpoint, not an old untracked training export'
            )

    legacy_geometry = {
        'module.instance_mask_estimator.xym_1024',
        'module.center_augmentator.xym',
        'module.instance_center_estimator.kernel_cos',
        'module.instance_center_estimator.kernel_sin',
    }
    for key in (values.keys() - expected.keys()) & legacy_geometry:
        del values[key]
    gaussian = 'module.instance_center_estimator.gaussian_blur.conv.weight'
    if gaussian in expected and gaussian not in values:
        values[gaussian] = expected[gaussian]
    center_model.load_state_dict(values, strict=True)


def load_center_model(args, state, device):
    center_model = get_center_model(args['center_model']['name'], center_model_kwargs(args), is_learnable=True)
    center_model.init_output(args['num_vector_fields'])
    center_model = torch.nn.DataParallel(center_model.to(device), device_ids=[0], dim=0)
    if state is not None:
        load_center_state(center_model, state)
    set_center_model_mode(center_model, training=False)
    return center_model
