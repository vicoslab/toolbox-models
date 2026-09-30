from label_studio_sdk.converter.brush import decode_rle
import numpy as np

def convert_point(width, height, value):
    x = int(value['x'] / 100 * width)
    y = int(value['y'] / 100 * height)
    return x, y

# todo: differentiate between grouped images, +when shared/not
def export(annotations, export_dir, relpaths, shared, config):
    if not annotations:
        return None
    points = []
    for tag in annotations[0] or []:
        kind, value, w, h = map(tag.__getitem__, ['type', 'value', 'original_width', 'original_height'])
        if kind == 'choices':
            continue
        elif kind == 'keypoint':
            points.append(convert_point(w, h, tag['value']))
        elif kind == 'vector':
            pts = []
            for vert in value['vertices']:
                pts.extend(convert_point(w, h, vert))
            points.append(pts)
        elif kind == 'rectangle':
            x = int((value['x'] + value['width']/2) / 100 * w)
            y = int((value['y'] + value['height']/2) / 100 * h)
            points.append([x, y])
        elif kind == 'brush':
            if rle := value.get('rle'):
                rgba = np.asarray(decode_rle(rle), dtype=np.uint8)
                mask = rgba.reshape(h, w, 4)[:, :, 3] > 0
                cy, cx = np.argwhere(mask).mean(axis=0)
                points.append([int(cx), int(cy)])
            else:
                print('Warning: unsupported brush format')

    # A present submission without points is a negative; no submission stays missing.
    return dict(points=points)