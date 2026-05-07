import os
import site
site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/super-simple-net')
import sys
import numpy as np
from model.supersimplenet import SuperSimpleNet
from torchvision.transforms.v2 import ToImage, Resize, ToDtype, Normalize, Compose
import torch
from PIL import Image, ImageOps

import modelargs

config = modelargs.parse('./model.json')

IMAGE_SIZE=(config['height'], config['width'])
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

model = SuperSimpleNet(image_size=IMAGE_SIZE, config=config).to(DEVICE)
if config['weights']:
    model.load_model(config['weights'])
model.eval()

transforms = Compose([
    ToImage(),
    Resize(size=IMAGE_SIZE),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
def transform(images: list[np.array]) -> torch.tensor:
    return torch.stack([transforms(image).to(DEVICE) for image in images])

def predict(tensor: torch.tensor, shapes: list[tuple[int,int]]) -> (list[np.array], np.array):
    anomaly_map, anomaly_score = model.forward(tensor)
    anomaly_map = [Resize(size=shape)(x).squeeze(0).sigmoid().numpy() for (x,shape) in zip(anomaly_map.detach().cpu(), shapes)]
    return anomaly_map, anomaly_score.detach().cpu().sigmoid().numpy()

def load(src):
    image = Image.open(src)
    image = ImageOps.exif_transpose(image)
    image = np.array(image.convert('RGB'))
    return image

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    image = load(sys.argv[1])

    maps, scores = predict(transform([image]), [image.shape[:2]])

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].title(scores[0])
    axs[1].imshow(maps[0])
    plt.savefig('test.png')
else:
    from label_studio_ml.api import init_app
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse
    from label_studio_sdk.converter import brush
    from typing import List, Dict, Optional
    from uuid import uuid4

    from flask import Flask, request
    from io import BytesIO
    import base64

    class SuperSimpleNet(LabelStudioMLBase):
        def get_results(self, maps, scores, width, height, from_name, to_name, label):
            results = []

            for anomaly_map, score in zip(maps, scores):

                mask = np.where(anomaly_map < 0.5, 0, 255)
                results.append({
                    'id': str(uuid4())[:6],
                    'from_name': from_name,
                    'to_name': to_name,
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0,
                    'value': {
                        'format': 'rle',
                        'rle': brush.mask2rle(mask),
                        'labels': [label],
                    },
                    'score': float(score),
                    'type': 'labels',
                    'readonly': False
                })

            return [{
                'result': results,
                'model_version': 'Super Simple Net',
            }]

        def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:

            from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Image')
            labels = None
            for tag_name, tag in self.parsed_label_config.items():
                if len(tag['labels']) > 0:
                    labels = tag['labels']
                    break
            
            images = list(map(lambda task: load(self.get_local_path(task['data'][value], task_id=task['id'])), tasks))
            maps, scores = predict(transform(images), [i.shape[:2] for i in images])
            predictions = self.get_results(
                maps=maps,
                scores=scores,
                width=IMAGE_SIZE[1],
                height=IMAGE_SIZE[0],
                from_name=from_name,
                to_name=to_name,
                label=labels[0])
            
            return ModelResponse(predictions=predictions)

    app = init_app(model_class=SuperSimpleNet)

    def encode(img):
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="WebP")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    @app.route('/infer', methods=["POST"])
    def infer():
        if 'images' not in request.files:
            return []
        
        images = list(map(load, request.files.getlist('images')))
        maps, scores = predict(transform(images), [x.shape[:2] for x in images])

        return {
            'anomaly_maps': list(map(encode, maps)),
            'scores': list(map(float, scores)),
        }
