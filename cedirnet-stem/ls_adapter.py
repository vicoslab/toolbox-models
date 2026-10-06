"""Label-driven Toolbox export for registered BF/HAADF STEM pairs.

Semantic regions become class-ID masks. Particle ellipses retain the mean pixel
radius; other filled regions become a centroid and equal-area circle. A region
is one instance, including disconnected brush strokes with the same region ID.
"""
import hashlib
import json
from pathlib import Path

from label_studio_sdk.converter.brush import decode_rle
import numpy as np
from PIL import Image
from stem_plugin.ls_geometry import normalize_results, shape_mask


def annotation_results(annotations):
    """Accept flat results or one Toolbox annotation; never merge annotators."""
    if not isinstance(annotations, list):
        raise ValueError('annotation results must be a list')
    if annotations and isinstance(annotations[0], list):
        if len(annotations) != 1:
            raise ValueError('export requires exactly one annotation per task')
        annotations = annotations[0]
    if any(not isinstance(tag, dict) or not isinstance(tag.get('value'), dict) for tag in annotations):
        raise ValueError('annotation results must contain region dictionaries')
    return annotations


def configured_labels(config):
    categories, classes, particles = {}, {}, {'Particle'}
    for control in config.iter():
        for node in control.findall('Label'):
            name = node.get('alias') or node.attrib['value']
            if node.get('category') is not None:
                category = int(node.attrib['category'])
                if not 0 <= category <= 255:
                    raise ValueError('semantic category must fit uint8')
                if name in categories or name in particles:
                    raise ValueError('label aliases must be unique across semantic and particle classes')
                categories[name] = category
                if category != 255:
                    if category in classes:
                        raise ValueError('semantic category IDs must be unique')
                    classes[category] = node.attrib['value']
            elif control.get('name') == 'labels':
                if name in categories:
                    raise ValueError('label aliases must be unique across semantic and particle classes')
                particles.add(name)
    if sorted(classes) != list(range(len(classes))):
        raise ValueError('semantic category IDs must be contiguous from zero')
    return categories, [classes[i] for i in sorted(classes)], particles


def region_mask(kind, value, width, height):
    if kind != 'brushlabels':
        return shape_mask(kind, value, width, height)
    if value.get('format', 'rle') != 'rle' or not value.get('rle'):
        raise ValueError('brush regions require RLE')
    rgba = np.asarray(decode_rle(value['rle']), dtype=np.uint8)
    if rgba.size != width * height * 4:
        raise ValueError('brush RLE length does not match original dimensions')
    return rgba.reshape(height, width, 4)[:, :, 3] > 0


def export(annotations, export_dir, relpaths, shared, config, **kwargs):
    categories, classes, particles = configured_labels(config)
    if len(relpaths) != 2:
        raise ValueError('STEM export requires a registered [BF, HAADF] image pair')
    if not annotations:
        return {}
    tags = annotation_results(annotations)
    points, regions, seen = [], [], set()
    dimensions = None
    for tag in normalize_results(tags):
        kind, value = tag.get('type'), tag.get('value', {})
        if kind == 'choices':
            continue
        if kind not in ('ellipselabels', 'rectanglelabels', 'polygonlabels', 'brushlabels', 'vectorlabels'):
            raise ValueError(f'unsupported annotation type {kind!r}')
        if tag.get('to_name', 'image') != 'image':
            raise ValueError('STEM regions must target image')
        if tag.get('item_index', 0) not in (0, 1):
            raise ValueError('STEM gallery item_index must be 0 or 1')
        size = (tag.get('original_width'), tag.get('original_height'))
        if any(type(x) is not int or x <= 0 for x in size):
            raise ValueError('regions require positive integer original dimensions')
        if tag.get('image_rotation', 0) != 0:
            raise ValueError('rotated image annotations must be reset before STEM export')
        if dimensions is not None and size != dimensions:
            raise ValueError('all regions in the registered image pair must have equal dimensions')
        dimensions = size
        w, h = size
        # Shared gallery copies are one region; distinct IDs remain distinct instances.
        signature = json.dumps([tag.get('id'), kind, tag.get('from_name'), value], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        labels = value.get(kind, [])
        # Legacy vectors are center/radius pairs and were exported without a class.
        legacy_vector = kind == 'vectorlabels' and not labels
        if not legacy_vector and (len(labels) != 1 or labels[0] not in categories.keys() | particles):
            raise ValueError('region requires one configured particle or semantic label/alias')
        semantic = not legacy_vector and labels[0] in categories
        if kind == 'vectorlabels':
            if semantic:
                raise ValueError('vectors cannot define filled semantic regions')
            vertices = value.get('vertices', [])
            if len(vertices) != 2:
                raise ValueError('particle vectors need exactly two vertices')
            coords = np.asarray([[v['x'], v['y']] for v in vertices], dtype=float)
            if not np.isfinite(coords).all() or (coords < 0).any() or (coords > 100).any():
                raise ValueError('particle vertices must be finite percentages in 0..100')
            coords *= [w / 100, h / 100]
            if coords[0, 0] >= w or coords[0, 1] >= h or np.linalg.norm(coords[1] - coords[0]) <= 0:
                raise ValueError('particle center must lie inside image and radius must be positive')
            points.append(coords.flatten().tolist())
        elif kind == 'ellipselabels' and not semantic:
            x, y, rx, ry, rotation = map(float, [value['x'], value['y'], value['radiusX'], value['radiusY'], value.get('rotation', 0)])
            if not np.isfinite([x, y, rx, ry, rotation]).all() or not (0 <= x < 100 and 0 <= y < 100 and 0 < rx <= 100 and 0 < ry <= 100):
                raise ValueError('particle ellipse requires finite center inside image and positive radii')
            points.append([x * w / 100, y * h / 100, (rx * w + ry * h) / 200])
        else:
            pixels = region_mask(kind, value, w, h)
            if not pixels.any():
                raise ValueError('filled region must contain at least one image pixel')
            if semantic:
                regions.append((categories[labels[0]], pixels))
            else:
                locations = np.argwhere(pixels)
                center = locations.mean(axis=0) + .5  # pixel-center coordinates
                radius = np.sqrt(len(locations) / np.pi)
                points.append([float(center[1]), float(center[0]), float(radius)])
    # Preserve Toolbox's submitted-empty annotation convention for particle negatives.
    result = {'points': points}
    if dimensions is not None:
        result['annotation_size'] = list(dimensions)
    if regions:
        w, h = dimensions
        mask = np.full((h, w), 255, dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        conflict = np.zeros((h, w), dtype=bool)
        ignored = np.zeros((h, w), dtype=bool)
        for category, pixels in regions:
            if category == 255:
                ignored |= pixels
                continue
            conflict |= occupied & pixels & (mask != category)
            mask[pixels & ~occupied] = category
            occupied |= pixels
        mask[conflict | ignored] = 255
        digest = hashlib.sha256(json.dumps(classes).encode() + repr(mask.shape).encode() + mask.tobytes()).hexdigest()
        relative = Path('semantic_masks') / f'{digest}.png'
        destination = Path(export_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(destination)
        result.update(semantic_mask=relative.as_posix(), semantic_classes=classes)
    return result
