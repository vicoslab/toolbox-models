"""Lossless semantic masks and Label Studio brush preannotations."""
import base64
import io
import colorsys
import numpy as np
from PIL import Image


def palette(classes):
    return [[round(c*255) for c in colorsys.hsv_to_rgb(i/max(len(classes),1), .7, .95)] for i in range(len(classes))]


def png_data(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream,format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def encode_mask(mask, classes):
    mask = np.asarray(mask, dtype=np.uint8)
    colors = palette(classes)
    rgba = np.zeros((*mask.shape,4),dtype=np.uint8)
    for i,color in enumerate(colors):
        rgba[mask==i] = [*color,120]
    return dict(width=mask.shape[1],height=mask.shape[0],classes=list(classes),
                colors=colors,mask_png=png_data(mask),overlay_png=png_data(rgba))


def brush_results(mask, classes, from_name, to_name):
    from label_studio_converter.brush import mask2rle
    h,w = mask.shape
    return [dict(id=f'semantic-{i}',from_name=from_name,to_name=to_name,
                 type='brushlabels',original_width=w,original_height=h,image_rotation=0,
                 value=dict(format='rle',rle=mask2rle((mask==i).astype(np.uint8)*255),brushlabels=[label]))
            for i,label in enumerate(classes) if np.any(mask==i)]
