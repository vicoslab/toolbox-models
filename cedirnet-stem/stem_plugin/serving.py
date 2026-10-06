"""Host-neutral inference and preannotation helpers (no eager model load)."""
import base64
import io
import os
import numpy as np
from PIL import Image
from .annotations import load_stem_image, load_single_stem_image
from .checkpoint import safe_torch_load
from .runtime import StemRuntime
from .task_options import task_config
from .results import label_studio_ellipse_result
from .semantic_results import brush_results
import torch


def load_runtime(options, device):
    tasks = task_config(options)
    path = options.get('model')
    if path:
        state = safe_torch_load(path,map_location=device)
    elif tasks.segmentation:
        raise ValueError('segmentation inference requires trained semantic weights')
    else:
        import torch
        url = "https://data.vicos.si/skokec/STEM/checkpoint.pth"
        print(f'Loading CeDiRNet-STEM model from "{url}"')
        state = torch.hub.load_state_dict_from_url(url,map_location=device)
    backbone = state.get('backbone',state.get('metadata',{}).get('backbone','tu-convnext_base'))
    runtime = StemRuntime(tasks,device,backbone,pretrained=False)
    if tasks.nanoparticles and options.get('localisation'):
        state = dict(state,center_model_state_dict=safe_torch_load(options['localisation'],map_location=device)['center_model_state_dict'])
    runtime.load(state,inference=True)
    return runtime.eval()


def load_pairs(files, modality=None):
    """Load paired batches, or one explicitly selected BF/HAADF detector."""
    if modality not in (None, 'paired', 'BF', 'HAADF'):
        raise ValueError('Input modality must be paired, BF or HAADF')
    if not files:
        raise ValueError('Provide at least one STEM image')
    if modality in ('BF', 'HAADF'):
        if len(files) != 1:
            raise ValueError('Single-modality inference requires exactly one image')
        return [np.asarray(load_single_stem_image(files[0], modality))]
    if len(files) == 1:
        raise ValueError('For a single image, select modality BF or HAADF explicitly')
    if len(files)%2:
        raise ValueError('Input images must be pairs of STEM images (first BF, then HAADF)')
    return [np.asarray(load_stem_image(*files[i:i+2])) for i in range(0,len(files),2)]


def _geometry_control(parsed_config, kind, target):
    """Find the shared tool for this image, not a tool for another target."""
    candidates = [name for name, tag in parsed_config.items()
                  if tag.get('type') == kind and tag.get('to_name') == [target]]
    if len(candidates) != 1:
        raise ValueError(f'Separate Labels predictions require exactly one {kind} control for {target!r}')
    return candidates[0]


def _region_results(result, tag, parsed_config, kind):
    """Separate controls serialize geometry plus Labels with identical IDs."""
    if any(image.get('valueList') for image in tag.get('inputs', [])):
        # index in preannotation is a batch index, not the BF/HAADF gallery index.
        result['item_index'] = 0
    if tag['type'] != 'Labels':
        return [result]
    tool = _geometry_control(parsed_config, kind, result['to_name'])
    geometry_value = {key: value for key, value in result['value'].items() if key != 'labels'}
    geometry = dict(result, from_name=tool, type=kind.lower(), value=geometry_value)
    return [geometry, dict(result, type='labels')]


def preannotation(response, index, size, parsed_config):
    """Resolve class controls and emit editor-loadable shared-tool regions."""
    results = []
    if response['tasks']['nanoparticles']:
        candidates = [(name, tag) for name, tag in parsed_config.items()
                      if tag.get('type') in ('Labels', 'EllipseLabels')
                      and tag.get('labels', []) == ['nanoparticle']]
        if len(candidates) != 1:
            raise ValueError('nanoparticles requires exactly one Labels or EllipseLabels control')
        name, tag = candidates[0]
        # Parsed labels are already serialized aliases, not display values (PtCo).
        labels = {tag['type'].lower(): tag['labels']}
        for j, (center, radius, score) in enumerate(zip(response['centers'][index], response['radii'][index], response['scores'][index])):
            result = label_studio_ellipse_result(center=center, radius=radius, score=score, original_size=size,
                from_name=name, to_name=tag['to_name'][0], label=labels, result_id=f'particle-{j}')
            results.extend(_region_results(result, tag, parsed_config, 'Ellipse'))
    semantic = response['segmentation'][index]
    if semantic is not None:
        labels = None
        for name, tag in parsed_config.items():
            # SDK keys are serialized aliases; model classes retain canonical values.
            aliases = {attrs.get('value', label): label
                    for label, attrs in tag.get('labels_attrs', {}).items()}
            _labels = [aliases.get(label, label) for label in semantic['classes']]
            if tag.get('type', '').endswith('Labels') and set(_labels).issubset(tag.get('labels', [])):
                tagid = tag['type'].lower()
                labels = [{tagid: [label]} for label in _labels]
                break
        if labels is None:
            raise ValueError('Labeling config does not include any *Labels tags with matching semantic classes.')
        mask = np.array(Image.open(io.BytesIO(base64.b64decode(semantic['mask_png'].split(',',1)[1]))))
        for result in brush_results(mask, labels, name, tag['to_name'][0]):
            results.extend(_region_results(result, tag, parsed_config, 'Brush'))
    scores = response['scores'][index]
    return dict(result=results,model_version='CeDiRNet-STEM-tasks-v2',score=sum(scores)/max(len(scores),1))
