import os
import site
site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/cedirnet/src')

import sys
from functools import partial
from typing import List, Tuple

import modelargs, json
from extras import plot_results, load_center_model

from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from tqdm import tqdm
from models import get_model
from utils.visualize.orientation import OrientationVisualizeTest


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cmd_args = modelargs.parse("model.json")

from base_config import args
# with open(cmd_args["config"],'r') as f:
#     args = json.load(f)

args['checkpoint_path'] = cmd_args["model"]
if path := cmd_args.get("localization_model", ""):
    args['center_checkpoint_path'] = path

model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['num_vector_fields'])
model = torch.nn.DataParallel(model).to(DEVICE)
model.eval()

print(f'Loading from "{args["checkpoint_path"]}"')
state = torch.load(args['checkpoint_path'], weights_only=False)
if 'model_state_dict' in state:
    if 'module.model.segmentation_head.2.weight' in state['model_state_dict']:
        checkpoint_input_weights = state['model_state_dict']['module.model.segmentation_head.2.weight']
        checkpoint_input_bias = state['model_state_dict']['module.model.segmentation_head.2.bias']
        model_output_weights = model.module.model.segmentation_head[2].weight
        if checkpoint_input_weights.shape != model_output_weights.shape:
            state['model_state_dict']['module.model.segmentation_head.2.weight'] = checkpoint_input_weights[:2, :, :, :]
            state['model_state_dict']['module.model.segmentation_head.2.bias'] = checkpoint_input_bias[:2]
            print('WARNING: #####################################################################################################')
            print('WARNING: regression output shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
            print('WARNING: #####################################################################################################')

    model.load_state_dict(state['model_state_dict'], strict=True)

if center_path := args.get('center_checkpoint_path'):
    print(f'Loading center model from "{center_path}"')
    center_model = load_center_model(args, torch.load(center_path, weights_only=False), DEVICE)
else:
    print("Loading center model from main model")
    center_model = load_center_model(args, state, DEVICE)

center_model = torch.nn.DataParallel(center_model).to(DEVICE)
center_model.eval()

def transform(images: List[np.array], size: Tuple[int, int]): # size should be (w,h)
    transformed = []
    for im in images:
        im = Image.fromarray(im).resize(size)
        im = np.array(im).transpose((2,0,1))
        transformed.append(torch.tensor(im))
    return torch.stack(transformed)

def to_float(x):
    return list(map(float, x))

def predict(images: List[np.array], size: Tuple[int, int]):
    tensor = transform(images, size)
    center_output = center_model(model(tensor), detect_centers=True)

    center_pred, angle_pred = map(lambda k: center_output[k].detach().cpu().numpy(), ['center_pred', 'pred_angle'])

    def get_result(input):
        center_pred, angle_pred = input
        valid = center_pred[:, 0] == 1

        scores = center_pred[valid, 4]
        desc_order = np.argsort(scores)[::-1]

        centers = list(map(to_float, center_pred[valid][desc_order, 1:4]))
        # convert returned coords in image space to relative ([0,1]), because resizing may have
        # occured and people might not be aware they need to rescale results
        for c in centers:
            c[0] /= size[0]
            c[1] /= size[1]
        scores = to_float(scores[desc_order])
        angles = to_float(angle_pred[valid][desc_order,:].flatten())

        return centers, scores, angles

    return zip(*map(get_result, zip(center_pred, angle_pred)))

def load(src):
    image = Image.open(src)
    image = ImageOps.exif_transpose(image)
    image = np.array(image.convert('RGB'))
    return image

if __name__ == "__main__":

    im = load(sys.argv[1])
    h, w, _ = im.shape
    w, h = (int(w // 32 * 32), int(h // 32 * 32))
    centers, scores, angles = predict([im], (w,h))

    fig, _ = plot_results(im, np.array(centers[0])*[w,h,0], scores[0], angles[0])
    fig.savefig('result.png')
else:
    from label_studio_ml.api import init_app
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse
    from label_studio_sdk.converter import brush
    from typing import List, Dict, Optional
    from uuid import uuid4

    from flask import Flask, request
    from werkzeug.utils import secure_filename
    from io import BytesIO
    import base64

    class CeDiRNet(LabelStudioMLBase):
        def get_results(self, centers, scores, angles, width, height, from_name, to_name, label, dist = 30):
            results = []

            for (x, y, _), score, angle in zip(centers, scores, np.deg2rad(angles)):
                dx, dy = np.cos(angle)*dist, np.sin(angle)*dist

                results.append({
                    'id': str(uuid4())[:6],
                    'from_name': from_name,
                    'to_name': to_name,
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0,
                    'value': {
                        'closed': False,
                        'vertices': [
                            { 'x': x, 'y': y, 'id': str(uuid4())[:21]},
                            { 'x': x+dx, 'y': y+dy, 'id': str(uuid4())[:21]},
                        ],
                        'labels': [label],
                    },
                    'score': float(score),
                    'type': 'labels',
                    'readonly': False
                })

            return [{
                'result': results,
                'model_version': 'CeDiRNet',
            }]

        def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:

            from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Image')
            labels = None
            for tag_name, tag in self.parsed_label_config.items():
                if len(tag['labels']) > 0:
                    labels = tag['labels']
                    break
            
            images = list(map(lambda task: load(self.get_local_path(task['data'][value], task_id=task['id'])), tasks))
            h, w, _ = images[0].shape
            w, h = (int(w // 32 * 32), int(h // 32 * 32))
            centers, scores, angles = predict(images, (w,h))
            predictions = self.get_results(
                centers=centers,
                scores=scores,
                angles=angles,
                width=IMAGE_SIZE[1],
                height=IMAGE_SIZE[0],
                from_name=from_name,
                to_name=to_name,
                label=labels[0])
            
            return ModelResponse(predictions=predictions)

    app = init_app(model_class=CeDiRNet)

    @app.route('/infer', methods=["POST"])
    def infer():
        if 'images' not in request.files:
            return []

        images = list(map(load, request.files.getlist('images')))
        h, w, _ = images[0].shape
        w, h = (int(w // 32 * 32), int(h // 32 * 32))
        centers, scores, angles = predict(images, (w,h))

        return {
            'centers': centers,
            'scores': scores,
            'angles': angles,
        }
