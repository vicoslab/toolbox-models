# so we parse -- and any possible future args?
import modelargs
_ = modelargs.parse("./model.json")

import os
os.chdir("../sam")

# import site
# site.addsitedir("/opt/apps/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image")
# from _wsgi import app

import torch
import numpy as np
import os
import sys
import pathlib
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from PIL import Image

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from label_studio_ml.api import init_app
from flask import request

DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT))

sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)

predictor = SAM2ImagePredictor(sam2_model)

class SegmentAnything2(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        results = []
        total_prob = 0
        for mask, prob in zip(masks, probs):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:4]
            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]

    def _sam_predict(self, point_coords=None, point_labels=None, input_box=None):
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])
        # logits = logits[sorted_ind]
        return {
            'masks': [mask],
            'probs': [prob]
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        from_name, to_name, value = self.get_first_tag_occurence('Brush', 'Image')
        if not context or not context.get('region'):
            # if there is no context, no interaction has happened yet
            return ModelResponse(predictions=[])

        image_path = self.get_local_path(tasks[0]['data'][value], task_id=tasks[0]['id'])
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)

        image_height, image_width, _ = image.shape

        # collect context information
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None

        region = context['region']
        x = int(region['x'] * image_width / 100)
        y = int(region['y'] * image_height / 100)
        ctx_type = region['type']
        selected_label = [x['value']['labels'][0] for x in region['results'] if 'labels' in x['value']][0]
        if ctx_type == 'keypointregion':
            point_labels.append(1 - int(region['negative']))
            point_coords.append([x, y])
        elif ctx_type == 'rectangleregion':
            box_width = region['width'] * image_width / 100
            box_height = region['height'] * image_height / 100
            input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        predictor_results = self._sam_predict(
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
        )

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label)
        
        return ModelResponse(predictions=predictions)

app = init_app(model_class=SegmentAnything2)

if __name__ == "__main__":
    app.run(host=args.host, port=args.port) #, debug=True)
