from label_studio_sdk.converter import brush
import numpy as np
from uuid import uuid4
import math

def export(annotations, export_dir, relpaths, shared):
    return export_(annotations, export_dir, relpaths, shared)[0]

# todo: differentiate between grouped images, +when shared/not
# convert label studio annotations into dataset samples
def export_(annotations, export_dir, relpaths, shared):
    points = []
    orig = []
    for tag in annotations[0] or []:
        res = {}
        value = tag['value']
        kind = tag.get('type')
        size = { 'x': tag['original_width'], 'y': tag['original_height'] }
        if kind == 'vector' and (verts := value.get('vertices')):
            if len(verts) != 2:
                print(f'Warning (skipping): vertices len != 2')
                continue
            orig.append(tag)
            points.append([int(v[p]/100*size[p]) for v in verts for p in ['x', 'y']])
        elif kind == 'ellipse':
            p1 = [int(value[p]/100*size[p]) for p in ['x', 'y']]
            r = int((value['radiusX']*size['x'] + value['radiusY']*size['y'])/2)
            p2 = [el - r if el > r else el + r for el in p1]
            orig.append(tag)
            points.append([*p1, *p2])
        elif rle := value.get('rle'): # type can be magic wand, labels, ...
            mask = np.reshape(brush.decode_rle(rle), [size['y'], size['x'], 4])[:, :, 3]
            locations = np.argwhere(mask)
            center = locations.mean(axis=0)
            perimiter = sorted(locations, key=lambda x: math.hypot(*(center - x)))
            orig.append(tag)
            points.append(list(map(int, [*perimiter[int(0.95*len(perimiter))], *center][::-1])))

    if len(points) == 0:
        return None, None
    return { 'points': points }, orig

# convert a single annotated dataset sample into label studio regions
def import_(data, originals):
    results = []
    for (cx, cy, rx, ry), orig in zip(data['points'], originals):
        r = math.hypot(cx - rx, cy - ry)
        width, height = orig['original_width'], orig['original_height']
        results.append({
            'id': str(uuid4())[:8],
            'from_name': orig['from_name'],
            'to_name': orig['to_name'],
            'original_width': width,
            'original_height': height,
            'value': {
                'x': cx / width * 100,
                'y': cy / height * 100,
                'radiusX': r / width * 100,
                'radiusY': r / height * 100,
                'labels': orig['value']['labels'], # TODO: this is hardcoded to reference a 'Labels' tag
            },
            'type': 'ellipse',
        })
    return results
