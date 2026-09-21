# so we parse -- and any possible future args?
import modelargs
_ = modelargs.parse("./model.json")

import os
import site
CACHE = os.environ["TOOLBOX_CACHE"]
os.chdir(f'{CACHE}/sam3')

import torch
import numpy as np
import sys
from PIL import Image, ImageOps

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MODEL_CHECKPOINT = f'{CACHE}/sam3/sam3.pt'

if torch.cuda.is_available():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def load(src):
    image = Image.open(src).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image

model = build_sam3_image_model(checkpoint_path=MODEL_CHECKPOINT, enable_inst_interactivity=True)
processor = Sam3Processor(model, confidence_threshold=0.5)

if __name__ == "__main__":
    pass
else:
    from typing import List, Dict, Optional
    from uuid import uuid4
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse
    from label_studio_sdk.converter import brush
    from io import BytesIO
    import json
    import base64
    from label_studio_ml.api import init_app
    from flask import request
    from sam3.model.sam3_tracker_utils import mask_to_box

    class SegmentAnything(LabelStudioMLBase):
        """Custom ML Backend model
        """

        def get_results(self, masks, probs, width, height, from_name, to_name, labels, extra):
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
                        **labels,
                        'format': 'rle',
                        'rle': rle,
                    },
                    'score': float(prob),
                    'type': 'labels',
                    'readonly': False,
                    **extra,
                })

            return [{
                'result': results,
                'model_version': self.get('model_version'),
                'score': total_prob / max(len(results), 1)
            }]

        def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
            """ Returns the predicted mask for a smart bbox that has been placed."""

            from_name, to_name, value = self.get_first_tag_occurence(('BrushLabels', 'Labels'), 'Image')

            if not context or not (regions := context.get('regions')):
                # if there is no context, no interaction has happened yet
                return ModelResponse(predictions=[])

            image_path = tasks[0]['data'][value]
            extra = {}
            if type(image_path) == list:
                idx = regions[0]['item_index']
                if any((region['item_index'] != idx for region in regions)):
                    return ModelResponse(predictions=[])
                image_path = image_path[idx]
                extra['item_index'] = idx

            image = load(self.get_local_path(image_path, task_id=tasks[0]['id']))
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)

            box = None
            points_coords = []
            points_labels = []
            image_width, image_height = image.size
            for region in regions:
                if region['type'] == 'rectangleregion':
                    x, y, box_width, box_height = [region[k] / 100 for k in ['x', 'y', 'width', 'height']]
                    box = [x + box_width/2, y + box_height/2, box_width, box_height]
                elif region['type'] == 'keypointregion':
                    x = int(region['x'] / 100 * image_width)
                    y = int(region['y'] / 100 * image_height)
                    pos = 1 - int(region.get('negative', False))
                    points_coords.append((x,y))
                    points_labels.append(pos)
                else:
                    pass

            if box is not None:
                inference_state = processor.add_geometric_prompt(state=inference_state, box=box, label=True)
                scores = inference_state['scores'].detach().cpu().type(torch.float32).numpy()
                masks = inference_state['masks'].detach().cpu().squeeze(1).numpy().astype(np.uint8)
            elif len(points_coords) > 0:
                masks, scores, _ = model.predict_inst(inference_state, point_coords=points_coords, point_labels=points_labels, multimask_output=False)
            else:
                return ModelResponse(predictions=[])

            sorted_ind = np.argsort(scores)[::-1]
            predictions = self.get_results(
                masks=masks[sorted_ind].astype(np.uint8),
                probs=scores[sorted_ind],
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                labels=context['labels'],
                extra=extra,
            )

            return ModelResponse(predictions=predictions)

    app = init_app(model_class=SegmentAnything)

    def encode(img):
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="WebP")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    @app.route('/infer', methods=["POST"])
    def index():
        if bbox := request.form.get('boxes'):
            bbox = json.loads(bbox)
        if 'image' not in request.files or not bbox or len(bbox) != 1:
            return { 'masks': [], 'scores': [] }
        
        image = load(request.files['image'])
        width, height = image.size

        x, y, box_width, box_height = bbox[0]
        box = np.array([x + box_width/2, y + box_height/2, box_width, box_height])

        inference_state = processor.set_image(image)
        #processor.reset_all_prompts(inference_state)
        inference_state = processor.add_geometric_prompt(state=inference_state, box=box, label=True)
        
        inference_state['scores'] = inference_state['scores'].detach().cpu().type(torch.float32).numpy()
        sorted_ind = np.argsort(inference_state['scores'])[::-1]
        boxes = list(mask_to_box(inference_state['masks'].detach()).cpu().squeeze(1).numpy()[sorted_ind])
        masks = list((inference_state['masks'].detach().cpu().squeeze(1).numpy()*255).astype(np.uint8)[sorted_ind])
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            masks[i] = masks[i][y1:y2, x1:x2]
            boxes[i] = [x1/width, y1/height, (x2-x1)/width, (y2-y1)/height]
        scores = inference_state['scores'][sorted_ind]

        return {
            'boxes': boxes,
            'masks': list(map(encode, masks)),
            'scores': scores.tolist(),
        }
