# todo: differentiate between grouped images, +when shared/not
def export(annotations, export_dir, relpaths, shared):
    points = None
    for tag in annotations[0] or []:
        if verts := tag['value'].get('vertices'):
            if len(verts) != 2:
                print(f'Warning (skipping): vertices len != 2')
                continue

            if points is None:
                points = []
                size = {
                    'x': tag['original_width'],
                    'y': tag['original_height'],   
                }
            
            points.append([int(v[p]/100*size[p]) for v in verts for p in ['x', 'y']])

    return points and { 'points': points }