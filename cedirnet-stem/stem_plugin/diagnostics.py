"""STEM adaptation of origin/cedirnet-3dof-visualization diagnostics.

Same deterministic train/validation, epoch-scoped MLflow process; circles replace
orientation arrows, BF/HAADF remain separate, semantic views use class IDs.
"""
import re
from pathlib import PurePosixPath
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch
from .semantic_results import palette


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def training_artifact_path(epoch, sample_name, subset):
    if subset not in {'training','validation'}:
        raise ValueError('invalid visualization subset')
    parts = PurePosixPath(str(sample_name).replace('\\','/')).parts
    safe = re.sub(r'[^A-Za-z0-9_-]+','_', '_'.join(p for p in parts if p not in {'..','.','/'})) or 'sample'
    return f'visualizations/epoch_{epoch+1:04d}/{subset}/{safe}-diagnostics.png'


def plot_training_diagnostics(*, image, centers=(), radii=(), scores=(),
        direction_output=None, localization_response=None, ground_truth_centers=(),
        semantic_target=None, semantic_prediction=None, classes=()):
    image = _numpy(image)
    if image.shape[0] == 3:
        image = image.transpose(1,2,0)
    particle = direction_output is not None
    semantic = semantic_prediction is not None
    count = 2 + (3 if particle else 0) + (2 if semantic else 0)
    fig,axes = plt.subplots(1,count,figsize=(4*count,4),constrained_layout=True,squeeze=False)
    axes = list(axes[0])
    axes[0].imshow(image[:,:,0],cmap='gray'); axes[0].set_title('BF + ground truth')
    axes[1].imshow(image[:,:,1],cmap='gray'); axes[1].set_title('HAADF')
    gt = _numpy(ground_truth_centers).reshape(-1,2)
    if len(gt):
        gt = gt[np.any(gt!=0,axis=1)]
        axes[0].scatter(gt[:,0],gt[:,1],marker='+',c='yellow')
    cursor = 2
    if particle:
        ax = axes[cursor]; ax.imshow(image[:,:,0],cmap='gray'); ax.set_title('Nanoparticles: center + radius')
        for center,radius,score in zip(_numpy(centers),_numpy(radii).reshape(-1),_numpy(scores).reshape(-1)):
            if np.isfinite(radius) and radius>0:
                ax.add_patch(Circle(center[:2],radius,fill=False,color='lime'))
                ax.text(*center[:2],f'{score:.2f}',color='lime',fontsize=6)
        direction = _numpy(direction_output)
        axes[cursor+1].imshow(np.arctan2(direction[0],direction[1]),cmap='hsv',vmin=-np.pi,vmax=np.pi)
        axes[cursor+1].set_title('Center-direction angle')
        axes[cursor+2].imshow(np.clip(_numpy(localization_response).squeeze(),0,1),vmin=0,vmax=1,cmap='viridis')
        axes[cursor+2].set_title('Localization response')
        cursor += 3
    if semantic:
        cmap = ListedColormap(np.array(palette(classes))/255)
        for ax,target,title in zip(axes[cursor:], [semantic_target,semantic_prediction],['Semantic ground truth','Semantic prediction']):
            if target is not None:
                mask = _numpy(target).squeeze()
                ax.imshow(np.ma.masked_where(mask==255,mask),cmap=cmap,vmin=-.5,vmax=len(classes)-.5,interpolation='nearest')
            ax.set_title(title)
        axes[-1].legend(handles=[Patch(color=cmap(i),label=c) for i,c in enumerate(classes)],loc='lower center',fontsize=7)
    for ax in axes: ax.axis('off')
    return fig
