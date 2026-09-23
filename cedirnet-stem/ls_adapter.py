import hashlib
import json
from pathlib import Path
from label_studio_sdk.converter.brush import decode_rle

import numpy as np
from PIL import Image
from stem_plugin.ls_geometry import normalize_results, shape_mask

def export(annotations, export_dir, relpaths, shared, config):
    categories = { x.attrib["value"]: int(x.attrib["category"]) for x in config.findall(".//Label[@category]") }
    if len(relpaths) != 2:
        raise ValueError('STEM export requires a registered [BF, HAADF] image pair')
    if len(annotations or []) != 1:
        raise ValueError('STEM export requires one exactly annotation per task')
    tags = annotations[0]
    points, brushes = [], []
    dimensions = None
    seen = set()
    for tag in tags:
        value = tag.get('value', {})
        kind = tag.get('type')
        size = (tag.get('original_width'), tag.get('original_height'))
        if dimensions is not None and size != dimensions:
            raise ValueError('all regions in the registered image pair must have equal dimensions')
        dimensions = size
        w, h = size
        if kind in ('ellipselabels', 'ellipse'):
            x, y, rx, ry, rotation = map(float, [value['x'], value['y'], value['radiusX'], value['radiusY'], value.get('rotation', 0)])
            points.append([x * w / 100, y * h / 100, (rx * w + ry * h) / 200])
        elif vertices := value.get('vertices'):
            if len(vertices) != 2:
                raise ValueError('particle vectors need exactly two vertices')
            coords = np.array([[v['x'], v['y']] for v in vertices], dtype=float)
            coords *= [w / 100, h / 100]
            points.append(coords.flatten().tolist())
        elif rle := value.get('rle'):
            label = None
            for k in value:
                if k.endswith("labels") and (items := value[k]):
                    for item in items:
                        if (id := categories.get(item)) is not None:
                            label = id
            rgba = np.asarray(decode_rle(value['rle']), dtype=np.uint8)
            mask = rgba.reshape(h, w, 4)[:, :, 3] > 0
            # no brush category -> particle
            if label is not None:
                brushes.append((label, mask))
            else:
                locations = np.argwhere(mask)
                center = locations.mean(axis=0)
                radius = np.sqrt(len(locations)/np.pi)
                points.append([*map(int, center[::-1]), int(radius)])
        else:
            print("Warning: skipped", tag)
    result = {}
    if dimensions is not None:
        result['annotation_size'] = list(dimensions)
    if points:
        result['points'] = points
    if brushes:
        w, h = dimensions
        mask = np.full((h, w), 255, dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        conflict = np.zeros((h, w), dtype=bool)
        for id, pixels in brushes:
            conflict |= occupied & pixels & (mask != id)
            mask[pixels & ~occupied] = id
            occupied |= pixels
        mask[conflict] = 255
        relative = (Path('semantic_masks') / relpaths[0]).with_suffix(".label.png")
        destination = Path(export_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(destination)
        classes = sorted([k for k, v in categories.items() if v != 255], key=lambda x: categories[x])
        result.update(semantic_mask=relative.as_posix(), semantic_classes=classes)
    return result
