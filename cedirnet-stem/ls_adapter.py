"""Toolbox v4 export adapter for registered BF/HAADF pairs.

One reviewed annotation per task; regions are shared across the aligned pair.
Class IDs follow config.yml (never region encounter order). Unpainted pixels and
cross-class overlaps are ignored. Explicit Ignore regions always win.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import yaml


def semantic_classes():
    view = ET.fromstring(yaml.safe_load(Path(__file__).with_name('config.yml').read_text())['config'])
    control = view.find(".//BrushLabels[@name='semantic']")
    if control is None:
        raise ValueError('config.yml needs semantic BrushLabels')
    classes = [node.attrib['value'] for node in control.findall('Label') if node.attrib['value'] != 'Ignore']
    if not 2 <= len(classes) <= 255 or len(set(classes)) != len(classes):
        raise ValueError('semantic labels must be 2..255 unique classes')
    return classes


def particle_labels():
    """SDK serializes alias when present; retain the old Particle export label."""
    view = ET.fromstring(yaml.safe_load(Path(__file__).with_name('config.yml').read_text())['config'])
    control = view.find(".//EllipseLabels[@name='labels']")
    if control is None or len(control.findall('Label')) != 1:
        raise ValueError('config.yml needs exactly one particle label')
    node = control.findall('Label')[0]
    return {node.get('alias') or node.attrib['value'], 'Particle'}


def export(annotations, export_dir, relpaths, shared):
    # shared is currently always False in Toolbox, including shared Image views.
    if len(relpaths) != 2:
        raise ValueError('STEM export requires a registered [BF, HAADF] image pair')
    if not annotations:
        return {}
    if len(annotations) != 1:
        raise ValueError('STEM export requires one reviewed annotation per task; resolve annotator conflicts first')
    tags = annotations[0]
    if not isinstance(tags, list):
        raise ValueError('annotation must be a list of results')
    classes = semantic_classes()
    accepted_particles = particle_labels()
    reviewed, points, brushes = set(), [], []
    negative_particles = False
    dimensions = None
    seen = set()
    for tag in tags:
        value = tag.get('value', {})
        if tag.get('from_name') == 'reviewed':
            reviewed.update(value.get('choices', []))
            continue
        if tag.get('from_name') == 'particle_review':
            negative_particles |= 'No nanoparticles' in value.get('choices', [])
            continue
        is_vector = 'vertices' in value
        is_ellipse = tag.get('type') == 'ellipselabels'
        is_brush = tag.get('type') == 'brushlabels'
        if not (is_vector or is_ellipse or is_brush):
            continue
        size = (tag.get('original_width'), tag.get('original_height'))
        if any(type(x) is not int or x <= 0 for x in size):
            raise ValueError('regions require positive integer original dimensions')
        if tag.get('image_rotation', 0) != 0:
            raise ValueError('rotated annotations must be reset before STEM export')
        if dimensions is not None and size != dimensions:
            raise ValueError('all regions in the registered image pair must have equal dimensions')
        dimensions = size
        # Shared-image exports may repeat the same region for both frames.
        signature = json.dumps([tag.get('from_name'), value], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        w, h = size
        if is_ellipse:
            labels = value.get('ellipselabels', [])
            if tag.get('from_name') != 'labels' or len(labels) != 1 or labels[0] not in accepted_particles:
                raise ValueError('particle ellipses require labels control and configured particle label/alias (or legacy Particle)')
            try:
                x, y, rx, ry, rotation = map(float, [value['x'], value['y'],
                    value['radiusX'], value['radiusY'], value.get('rotation', 0)])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('particle ellipse requires numeric center and radii') from exc
            if not np.isfinite([x, y, rx, ry, rotation]).all():
                raise ValueError('particle ellipse geometry must be finite')
            if not (0 <= x < 100 and 0 <= y < 100 and 0 < rx <= 100 and 0 < ry <= 100):
                raise ValueError('particle ellipse center must be inside image and radii positive percentages')
            # Label Studio stores ellipse CENTER, not the bounding-box corner.
            # Keep the existing circle target: arithmetic mean of pixel radii.
            points.append([x * w / 100, y * h / 100, (rx * w + ry * h) / 200])
        elif is_vector:
            labels = value.get('vectorlabels', value.get('labels', []))
            if labels and (len(labels) != 1 or labels[0] not in accepted_particles):
                raise ValueError('particle vectors must have configured particle label/alias (or legacy Particle)')
            vertices = value['vertices']
            if len(vertices) != 2:
                raise ValueError('particle vectors need exactly two vertices')
            coords = np.array([[v['x'], v['y']] for v in vertices], dtype=float)
            if not np.isfinite(coords).all() or (coords < 0).any() or (coords > 100).any():
                raise ValueError('particle vertices must be finite percentages in 0..100')
            coords *= [w / 100, h / 100]
            if coords[0, 0] >= w or coords[0, 1] >= h or np.linalg.norm(coords[1] - coords[0]) <= 0:
                raise ValueError('particle center must lie inside image and radius must be positive')
            points.append(coords.flatten().tolist())
        else:
            if tag.get('from_name') != 'semantic':
                raise ValueError('brush control must be semantic')
            labels = value.get('brushlabels', [])
            if len(labels) != 1 or labels[0] not in [*classes, 'Ignore']:
                raise ValueError('unknown semantic brush label; match config.yml class order')
            if value.get('format') != 'rle':
                raise ValueError('semantic brushes require RLE')
            # Export host ships this SDK; standalone training env ships converter.
            try:
                from label_studio_sdk.converter.brush import decode_rle
            except ImportError:
                from label_studio_converter.brush import decode_rle
            rgba = np.asarray(decode_rle(value['rle']), dtype=np.uint8)
            if rgba.size != w * h * 4:
                raise ValueError('brush RLE length does not match original dimensions')
            brushes.append((labels[0], rgba.reshape(h, w, 4)[:, :, 3] > 0))
    result = {}
    if dimensions is not None:
        result['annotation_size'] = list(dimensions)
    if negative_particles and points:
        raise ValueError('explicit particle negative conflicts with particle regions')
    # Task visibility is NOT supervision. Only explicit review confirms negatives.
    # Legacy reviewed choices remain supported; ambiguous empty submissions do not.
    if points or 'Nanoparticles' in reviewed or negative_particles:
        result['points'] = points
    if brushes:
        w, h = dimensions
        mask = np.full((h, w), 255, dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)
        conflict = np.zeros((h, w), dtype=bool)
        ignored = np.zeros((h, w), dtype=bool)
        for name, pixels in brushes:
            if name == 'Ignore':
                ignored |= pixels
                continue
            class_id = classes.index(name)
            conflict |= occupied & pixels & (mask != class_id)
            mask[pixels & ~occupied] = class_id
            occupied |= pixels
        mask[conflict | ignored] = 255
        # Content addressing avoids overwrites when COMBINE reuses an export dir.
        digest = hashlib.sha256(json.dumps(classes).encode() + mask.shape.__repr__().encode() + mask.tobytes()).hexdigest()
        relative = Path('semantic_masks') / f'{digest}.png'
        destination = Path(export_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(destination)
        result.update(semantic_mask=relative.as_posix(), semantic_classes=classes)
    elif 'Segmentation' in reviewed:
        raise ValueError('reviewed segmentation needs at least one brush (use Ignore for unlabelled pixels)')
    return result
