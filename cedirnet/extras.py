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
    # ax.scatter(*zip(*centers[:,:2]), c='lime', marker='+')
    for (x, y, _), score, angle in zip(centers, scores, np.deg2rad(angles)):
        dx, dy = np.cos(angle)*dist, np.sin(angle)*dist
        ax.annotate('', xytext=(x, y), xy=(x+dx, y+dy), arrowprops=dict(color='lime', arrowstyle='->'))
        ax.annotate(f'{score:.2f}', xy=(x-dx/4, y-dy/4), size='xx-small', ha='center', va='center', c='lime', path_effects=[
            SimpleLineShadow(shadow_color="black", linewidth=1, offset=(0,0), alpha=0.7),
            Normal()
        ])
    return fig, ax

# center model likely supports 6dof so we may need to load only a subset of weights
def load_center_model(args, state, device):
    center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'], is_learnable=True)
    center_model.init_output(args['loss_opts']['num_vector_fields'])
    center_model = torch.nn.DataParallel(center_model.to(device), device_ids=[0], dim=0)

    if state is None:
        return center_model

    INPUT_WEIGHTS_KEY = 'module.instance_center_estimator.conv_start.0.weight'

    if (checkpoint_input_weights := state['center_model_state_dict'].get(INPUT_WEIGHTS_KEY)) is not None:

        center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
        if checkpoint_input_weights.shape != center_input_weights.shape:
            state['center_model_state_dict'][INPUT_WEIGHTS_KEY] = checkpoint_input_weights[:, :2, :, :]

    center_model.load_state_dict(state['center_model_state_dict'], strict=False)
    return center_model
