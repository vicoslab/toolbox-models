#! .venv/bin/python

import sys
import numpy as np
from model.supersimplenet import SuperSimpleNet
from torchvision.transforms.v2 import ToImage, Resize, ToDtype, Normalize, Compose
import torch

import argparse_from_jsonschema

try:
    i = sys.argv.index('--')
    old = sys.argv[:i]
    sys.argv = [sys.argv[0]] + sys.argv[i+1:]
    config = argparse_from_jsonschema.parse(schema='./model.json')
    sys.argv = old
except:
    config = argparse_from_jsonschema.parse(schema='./model.json')

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

def predict(tensor: torch.tensor) -> list[np.array]:
    anomaly_map, anomaly_score = model.forward(tensor)
    anomaly_map = list(anomaly_map.detach().cpu().sigmoid().numpy().transpose(0,2,3,1))
    return anomaly_map, anomaly_score.detach().cpu().sigmoid().numpy()

if __name__ == '__main__':
    from PIL import Image, ImageOps
    from matplotlib import pyplot as plt

    image = Image.open(sys.argv[1])
    image = ImageOps.exif_transpose(image)
    image = np.array(image.convert('RGB'))

    anomaly_map, anomaly_score = predict(transform([image]))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].imshow(anomaly_map[0])
    plt.savefig('test.png')
else:
    from flask import Flask

    app = Flask(__name__)

    @app.route('/')
    def index():
        return '<span style="color:red">I am app 1</span>'
