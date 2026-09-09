from PIL import Image
import numpy as np
from label_studio_sdk.converter import brush

# todo: differentiate between grouped images, +when shared/not
def export(annotations, export_dir, relpaths, shared):
    mask = width = height = None
    for tag in annotations[0] or []:
        value = tag['value']
        if rle := value.get('rle'):
            if mask is None:
                width = tag['original_width']
                height = tag['original_height']
                mask = np.zeros((height, width), dtype=np.uint8)
            mask += np.reshape(brush.decode_rle(rle), [height, width, 4])[:, :, 3]

    if mask is None:
        return dict(label='normal')
    
    filename = (export_dir / relpaths[0]).with_suffix('.label.png')
    filename.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(filename)
    return dict(label='abnormal' if mask.sum() > 0 else 'normal', mask_path=str(filename.relative_to(export_dir)))