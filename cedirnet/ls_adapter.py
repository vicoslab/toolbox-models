
def convert_point(width, height, value):
    x = int(value['x'] / 100 * width)
    y = int(value['y'] / 100 * height)
    return x, y

# todo: differentiate between grouped images, +when shared/not
def export(annotations, export_dir, relpaths, shared):
    points = []
    for tag in annotations[0] or []:
        value = tag['value']
        kind = tag['type']
        if kind == 'keypoint':
            points.append(convert_point(tag['original_width'], tag['original_height'], tag['value']))
        elif kind == 'vector':
            pts = []
            w, h = tag['original_width'], tag['original_height']
            for vert in value['vertices']:
                pts.extend(convert_point(w, h, vert))
            points.append(pts)
        # todo: from segmentation/rectangle

    if not points:
        return None

    return dict(points=points)