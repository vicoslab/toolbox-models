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

def predict(tensor: torch.tensor) -> list[np.array]:
    anomaly_map, anomaly_score = model.forward(tensor)
    anomaly_map = list(anomaly_map.detach().cpu().squeeze(1).sigmoid().numpy())
    return anomaly_map, anomaly_score.detach().cpu().sigmoid().numpy()

def load(src):
    image = Image.open(src)
    image = ImageOps.exif_transpose(image)
    image = np.array(image.convert('RGB'))
    return image

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    image = load(sys.argv[1])

    maps, scores = predict(transform([image]))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].title(scores[0])
    axs[1].imshow(maps[0])
    plt.savefig('test.png')
else:
    from flask import Flask, request
    from io import BytesIO
    import base64

    app = Flask(__name__)

    def encode(img):
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="WebP")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    @app.route('/', methods=["POST"])
    def index():
        if 'images' not in request.files:
            return []
        
        images = map(load, request.files.getlist('images'))
        maps, scores = predict(transform(images))

        return {
            'anomaly_maps': list(map(encode, maps)),
            'scores': list(map(float, scores)),
        }
