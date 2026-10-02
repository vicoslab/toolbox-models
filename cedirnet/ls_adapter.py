from label_studio_sdk.converter.brush import decode_rle
import numpy as np

def convert_point(width, height, value):
    x = int(value['x'] / 100 * width)
    y = int(value['y'] / 100 * height)
    return x, y

def annotation_results(annotations):
    """Accept flat results or one Toolbox annotation; never merge annotators."""
    if not isinstance(annotations, list):
        raise ValueError('annotation results must be a list')
    if annotations and isinstance(annotations[0], list):
        if len(annotations) != 1:
            raise ValueError('export requires exactly one annotation per task')
        annotations = annotations[0]
    if any(not isinstance(tag, dict) for tag in annotations):
        raise ValueError('annotation results must contain region dictionaries')
    return annotations


# todo: differentiate between grouped images, +when shared/not
def export(annotations, export_dir, relpaths, shared, config=None, **kwargs):
    if not annotations:
        return None
    points = []
    for tag in annotation_results(annotations):
        kind = tag['type']
        if kind == 'choices':
            continue
        value, w, h = map(tag.__getitem__, ['value', 'original_width', 'original_height'])
        if kind in ('keypointlabels', 'keypoint'):
            points.append(convert_point(w, h, tag['value']))
        elif kind in ('vectorlabels', 'vector'):
            pts = []
            for vert in value['vertices']:
                pts.extend(convert_point(w, h, vert))
            points.append(pts)
        elif kind in ('rectanglelabels', 'rectangle'):
            x = int((value['x'] + value['width']/2) / 100 * w)
            y = int((value['y'] + value['height']/2) / 100 * h)
            points.append([x, y])
        elif kind in ('brushlabels', 'brush', 'magicwand'):
            if rle := value.get('rle'):
                rgba = np.asarray(decode_rle(rle), dtype=np.uint8)
                mask = rgba.reshape(h, w, 4)[:, :, 3] > 0
                cy, cx = np.argwhere(mask).mean(axis=0)
                points.append([int(cx), int(cy)])
            else:
                print('Warning: unsupported brush format')

    # A present submission without points is a negative; no submission stays missing.
    return dict(points=points)