"""Label Studio image results, including separate geometry/Labels controls.

LS 1.23 editor serializes Magicwand as RGBA RLE with a separate brushlabels
result sharing its region id when a semantic class is assigned. Shape coordinates are percentages; rotations are
clockwise in image pixels. Raster masks sample pixel centers.
"""
import copy
import numpy as np

KINDS = ('ellipse', 'rectangle', 'polygon', 'brush')


def normalize_results(tags):
    """Join separate Labels by region id, target and gallery frame, not order."""
    labels = {}
    for tag in tags:
        if tag.get('type') != 'labels':
            continue
        key = (tag.get('id'), tag.get('to_name'), tag.get('item_index', 0))
        if not key[0]:
            raise ValueError('separate labels require a region id')
        if key in labels and labels[key]['value'] != tag['value']:
            raise ValueError('conflicting labels for one region')
        labels[key] = tag
    consumed = set()
    for raw in tags:
        tag = copy.deepcopy(raw)
        kind = tag.get('type', '')
        if kind == 'magicwand':
            key = (tag.get('id'), tag.get('to_name'), tag.get('item_index', 0))
            paired = [t for t in tags if t.get('type') == 'brushlabels' and
                      (t.get('id'), t.get('to_name'), t.get('item_index', 0)) == key]
            if not key[0] or not paired:
                raise ValueError('Magicwand region has no semantic label; assign a class before re-export')
            for other in paired:
                for field in ('original_width', 'original_height', 'image_rotation'):
                    if other.get(field, 0) != tag.get(field, 0):
                        raise ValueError('Magicwand and label dimensions/rotation disagree')
                if any(other.get('value', {}).get(f) != tag.get('value', {}).get(f) for f in ('format', 'rle')):
                    raise ValueError('Magicwand and label masks disagree')
            # The paired brushlabels contains the same geometry and its class.
            continue
        if kind == 'labels':
            continue
        base = kind.removesuffix('labels')
        if base in KINDS:
            value = tag.setdefault('value', {})
            key = (tag.get('id'), tag.get('to_name'), tag.get('item_index', 0))
            paired = labels.get(key)
            own = value.get(base+'labels')
            if paired:
                for field in ('original_width', 'original_height', 'image_rotation'):
                    if field in paired and paired[field] != tag.get(field, 0):
                        raise ValueError('geometry and labels dimensions/rotation disagree')
                for field in value.keys() & paired['value'].keys():
                    if value[field] != paired['value'][field]:
                        raise ValueError('geometry and labels geometry disagree')
                incoming = paired['value'].get('labels', [])
                if own is not None and own != incoming:
                    raise ValueError('conflicting geometry and separate labels')
                value[base+'labels'] = incoming
                tag['from_name'] = paired.get('from_name')
                consumed.add(key)
            if not value.get(base+'labels'):
                raise ValueError(f'{base} region {tag.get("id")!r} has no label; use the configured particle or semantic label and re-export with its Labels result')
            tag['type'] = base+'labels'
        yield tag
    if labels.keys() - consumed:
        raise ValueError('labels without matching geometry (check region id, target and item_index)')


def shape_mask(kind, value, width, height):
    """Rasterize supported filled shapes; no implicit background or holes."""
    yy, xx = np.mgrid[:height, :width].astype(float)
    xx += .5
    yy += .5
    try:
        if kind == 'polygonlabels':
            if value.get('closed', True) is not True:
                raise ValueError('semantic polygon must be closed')
            points = np.asarray(value['points'], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                raise ValueError('polygon requires at least three xy points')
            if not np.isfinite(points).all() or (points < 0).any() or (points > 100).any():
                raise ValueError('polygon points must be finite percentages in 0..100')
            points = points * [width/100, height/100]
            area = np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1))
            if abs(area) < 1e-10:
                raise ValueError('polygon must have nonzero area')
            mask = np.zeros((height,width),dtype=bool)
            for (x1,y1),(x2,y2) in zip(points,np.roll(points,-1,axis=0)):
                if y1 != y2:
                    mask ^= ((y1 > yy) != (y2 > yy)) & (xx < (x2-x1)*(yy-y1)/(y2-y1)+x1)
            return mask
        x,y,rotation = map(float, (value['x'], value['y'], value.get('rotation',0)))
        fields = ('radiusX','radiusY') if kind == 'ellipselabels' else ('width','height')
        a,b = (float(value[f]) for f in fields)
        if not np.isfinite([x,y,rotation,a,b]).all():
            raise ValueError('shape geometry must be finite')
        if not (0 <= x <= 100 and 0 <= y <= 100 and 0 < a <= 100 and 0 < b <= 100):
            raise ValueError('shape coordinates require percentages and positive extents')
        dx,dy = xx-x*width/100, yy-y*height/100
        angle=np.deg2rad(rotation)
        u=dx*np.cos(angle)+dy*np.sin(angle)
        v=-dx*np.sin(angle)+dy*np.cos(angle)
        a,b=a*width/100,b*height/100
        if kind == 'ellipselabels':
            return (u/a)**2+(v/b)**2 <= 1
        return (u>=0)&(u<a)&(v>=0)&(v<b)
    except (KeyError,TypeError) as exc:
        raise ValueError(f'invalid {kind} geometry') from exc
