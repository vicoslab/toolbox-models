# so we parse -- and any possible future args?
import modelargs
_ = modelargs.parse("./model.json")

import os
os.chdir("../sam")
import site
CACHE = os.environ["TOOLBOX_CACHE"]
site.addsitedir(f'{CACHE}/sam-ls-backend')

import torch
import numpy as np
import sys
from PIL import Image, ImageOps

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MODEL_CHECKPOINT = f'{CACHE}/sam/sam3.pt'

if torch.cuda.is_available():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def load(src):
    image = Image.open(src)
    image = ImageOps.exif_transpose(image)
    return image

model = build_sam3_image_model(checkpoint_path=MODEL_CHECKPOINT)
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

    class SegmentAnything(LabelStudioMLBase):
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
                        'labels': [label],
                    },
                    'score': float(prob),
                    'type': 'labels',
                    'readonly': False
                })

            return [{
                'result': results,
                'model_version': self.get('model_version'),
                'score': total_prob / max(len(results), 1)
            }]

        def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
            """ Returns the predicted mask for a smart bbox that has been placed."""

            from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Image')
            labels = None
            for tag_name, tag in self.parsed_label_config.items():
                if len(tag['labels']) > 0:
                    labels = tag['labels']
                    break

            if not context or not context.get('region'):
                # if there is no context, no interaction has happened yet
                return ModelResponse(predictions=[])

            image = load(self.get_local_path(tasks[0]['data'][value], task_id=tasks[0]['id']))
            inference_state = processor.set_image(image)

            image_width, image_height = image.size

            region = context['region']
            if region['type'] != 'rectangleregion':
                return ModelResponse(predictions=[])

            x, y, box_width, box_height = [region[k] / 100 for k in ['x', 'y', 'width', 'height']]
            box = [x + box_width/2, y + box_height/2, box_width, box_height]

            processor.reset_all_prompts(inference_state)
            inference_state = processor.add_geometric_prompt(state=inference_state, box=box, label=True)

            inference_state['scores'] = inference_state['scores'].detach().cpu().type(torch.float32).numpy()
            sorted_ind = np.argsort(inference_state['scores'])[::-1].copy()
            inference_state['masks'] = inference_state['masks'][sorted_ind].detach().cpu().squeeze(1).numpy().astype(np.uint8)
            inference_state['scores'] = inference_state['scores'][sorted_ind]

            predictions = self.get_results(
                masks=inference_state['masks'],
                probs=inference_state['scores'],
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=labels[0])

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
        masks = (inference_state['masks'].detach().cpu().squeeze(1).numpy()*255).astype(np.uint8)[sorted_ind]
        scores = inference_state['scores'][sorted_ind]

        return {
            'masks': list(map(encode, masks)),
            'scores': list(map(float, scores)),
        }
