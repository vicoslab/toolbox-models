"""Host-neutral inference and preannotation helpers (no eager model load)."""
import base64
import io
import os
import numpy as np
from PIL import Image
from .annotations import load_stem_image
from .checkpoint import safe_torch_load
from .runtime import StemRuntime
from .task_options import task_config
from .results import label_studio_ellipse_result
from .semantic_results import brush_results


def load_runtime(options, device):
    tasks = task_config(options)
    path = options.get('model')
    if not path:
        if tasks.segmentation:
            raise ValueError('segmentation inference requires trained semantic weights')
        path = os.path.join(os.environ['TOOLBOX_CACHE'],'cedirnet-stem','stem_checkpoint.pt')
    state = safe_torch_load(path,map_location=device)
    backbone = state.get('backbone',state.get('metadata',{}).get('backbone','tu-convnext_base'))
    runtime = StemRuntime(tasks,device,backbone,pretrained=False)
    if tasks.nanoparticles and options.get('localisation'):
        state = dict(state,center_model_state_dict=safe_torch_load(options['localisation'],map_location=device)['center_model_state_dict'])
    runtime.load(state,inference=True)
    return runtime.eval()


def load_pairs(files):
    if not files or len(files)%2:
        raise ValueError('Input images must be pairs of STEM images (first BF, then HAADF)')
    return [np.asarray(load_stem_image(*files[i:i+2])) for i in range(0,len(files),2)]


def preannotation(response, index, size, parsed_config):
    """Resolve controls by type, never use a semantic label as a particle label."""
    results = []
    if response['tasks']['nanoparticles']:
        candidates = [(name,tag) for name,tag in parsed_config.items() if tag.get('type','').lower() == 'ellipselabels']
        if len(candidates)!=1:
            raise ValueError('nanoparticles requires exactly one EllipseLabels control')
        name,tag = candidates[0]
        labels = tag.get('labels', [])
        if len(labels) != 1:
            raise ValueError('particle control must contain exactly one configured label/alias; defect classification is semantic')
        # The SDK parsed config already resolves Label alias before value.
        label = labels[0]
        for j,(center,radius,score) in enumerate(zip(response['centers'][index],response['radii'][index],response['scores'][index])):
            results.append(label_studio_ellipse_result(center=center,radius=radius,score=score,original_size=size,
                from_name=name,to_name=tag['to_name'][0],label=label,result_id=f'particle-{j}'))
    semantic = response['segmentation'][index]
    if semantic is not None:
        candidates = [(name,tag) for name,tag in parsed_config.items() if tag.get('type','').lower()=='brushlabels']
        if len(candidates)!=1:
            raise ValueError('segmentation requires exactly one BrushLabels control')
        name,tag = candidates[0]
        if not set(semantic['classes']).issubset(tag.get('labels',[])):
            raise ValueError('BrushLabels must include all configured semantic classes')
        mask = np.array(Image.open(io.BytesIO(base64.b64decode(semantic['mask_png'].split(',',1)[1]))))
        results.extend(brush_results(mask,semantic['classes'],name,tag['to_name'][0]))
    scores = response['scores'][index]
    return dict(result=results,model_version='CeDiRNet-STEM-tasks-v2',score=sum(scores)/max(len(scores),1))
