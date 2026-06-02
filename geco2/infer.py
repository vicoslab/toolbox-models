import os
import site
site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/geco2')
import sys
import torch
from torch.nn import DataParallel
from models.counter_infer import CNT
import modelargs
from utils.data import resize_and_pad
import torchvision.ops as ops
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = modelargs.parse('./model.json')
args = dict(
    zero_shot = True,
    image_size = 1024,
    num_objects = 3,
    emb_dim = 256,
    kernel_dim = 3,
    reduction = 16,
    **args,
)
model = DataParallel(CNT(**args).to(device))
model.load_state_dict(torch.load(f'{os.environ["TOOLBOX_CACHE"]}/geco2/CNTQG_multitrain_ca44.pt', weights_only=True)['model'], strict=False)
model.eval()

def load(src):
    image = Image.open(src)
    image = ImageOps.exif_transpose(image)
    image = np.array(image.convert('RGB'))
    return image

transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
# bboxes should be [[x_topl, y_topl, x_botr, y_botr], ...]
def transform(f):
    def wrapped(image: np.array, bboxes: np.array, enable_mask):
        image_tensor = transforms(image)
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        img, boxes, scale = resize_and_pad(image_tensor, bboxes_tensor, size=args['image_size'])
        resize_bboxes = lambda boxes: (boxes / scale * args['image_size']).tolist()
        
        if not enable_mask:
            pred_boxes, pred_scores = f(img, boxes, enable_mask)
            return resize_bboxes(pred_boxes), pred_scores.tolist()
        
        pred_boxes, pred_scores, masks = f(img, boxes, enable_mask)
        masks = T.Resize((int(args['image_size'] / scale),)*2, interpolation=T.InterpolationMode.NEAREST)(masks)[:, :image.shape[0], :image.shape[1]]
        return resize_bboxes(pred_boxes), pred_scores.tolist(), np.array(masks)
    return wrapped

@transform
def predict(img, bboxes, enable_mask):
    model.module.return_masks = enable_mask

    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _, _, masks = model(img, bboxes)

    pred_boxes, box_v = [outputs[0][k][0].detach().cpu() for k in ['pred_boxes', 'box_v']]

    keep = ops.nms(pred_boxes, box_v, 0.5)
    pred_boxes = torch.clamp(pred_boxes[keep], 0, 1)
    box_v = box_v[keep]

    if enable_mask:
        return pred_boxes, box_v, masks[0].detach().cpu()[keep]

    return pred_boxes, box_v

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from PIL import ImageDraw, ImageFont
    image = load(sys.argv[1])
    drawn_boxes = [[100, 100, 200, 200]]

    pred_boxes, scores, masks = predict(image, drawn_boxes, True)

    N_masks = masks.shape[0]
    indices = torch.randint(1, N_masks + 1, (1, N_masks)).view(-1, 1, 1)
    mask_display = (masks * indices).sum(dim=0) # [H, W]
    
    rgba = plt.cm.tab20(plt.Normalize(vmin=0, vmax=N_masks)(mask_display))
    rgba[:, :, -1] = np.where(mask_display == 0, 0, 0.5)
    overlay = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    image_pil = Image.fromarray(image).convert('RGBA')
    image_pil = Image.alpha_composite(image_pil, overlay)

    draw = ImageDraw.Draw(image_pil)
    for box in pred_boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline='orange', width=2)
    for box in drawn_boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=3)

    w, h = image_pil.size
    sq = int(0.05 * w)
    x1, y1 = 10, h - sq - 10
    draw.rectangle([x1, y1, x1+sq, y1+sq], outline='black', fill='black')
    font = ImageFont.load_default()
    txt = str(len(pred_boxes))
    text_x = x1 + (sq - draw.textlength(txt, font=font)) / 2
    text_y = y1 + (sq - 10) / 2
    draw.text((text_x, text_y), txt, fill='white', font=font)

    image_pil.save('result.png')
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

    class GeCo2(LabelStudioMLBase):
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
            region = context['region']
            if region['type'] != 'rectangleregion':
                return ModelResponse(predictions=[])

            image = load(self.get_local_path(tasks[0]['data'][value], task_id=tasks[0]['id']))
            inference_state = processor.set_image(image)

            image_width, image_height = image.size

            x, y, box_width, box_height = [region[k] / 100 for k in ['x', 'y', 'width', 'height']]
            # geco2 expects topleft and bottomright corners in absolute
            box = [x * image_width, y * image_height, (x + box_width) * image_width, (y + box_height) * image_height]

            _, scores, masks = predict(image, [box], True)

            predictions = self.get_results(
                masks=masks,
                probs=scores,
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=labels[0])

            return ModelResponse(predictions=predictions)

    app = init_app(model_class=GeCo2)

    def encode(img):
        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="WebP")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    @app.route('/infer', methods=['POST'])
    def index():
        if 'image' not in request.files or 'bbox' not in request.form:
            return []
        
        image = load(request.files['image'])
        image_height, image_width, _ = image.shape

        x, y, box_width, box_height = json.loads(request.form['bbox'])
        box = [x * image_width, y * image_height, (x + box_width) * image_width, (y + box_height) * image_height]

        boxes, scores, masks = predict(image, [box], True)

        return {
            'boxes': boxes,
            'masks': list(map(encode, masks)),
            'scores': scores,
        }
