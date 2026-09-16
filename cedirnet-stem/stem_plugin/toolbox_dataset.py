"""Strict paired-image manifest adapter; class IDs are never binarized."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .annotations import load_stem_image, build_targets


class ToolboxDataset(Dataset):
    def __init__(self, manifest, tasks, split='train', size=(512,512), augment=False,
                 max_num_centers=2048, allow_empty=False):
        self.root = Path(manifest).resolve().parent
        data = json.loads(Path(manifest).read_text())
        if data.get('version',0) < 2:
            raise ValueError('manifest version must be >= 2')
        self.items = data.get(split, data.get('data',[]) if split == 'train' else [])
        if not self.items and not allow_empty:
            raise ValueError(f'manifest split {split!r} is empty')
        for item in self.items:
            if tasks.segmentation and item.get('semantic_classes', data.get('semantic_classes')) != list(tasks.classes):
                raise ValueError('item semantic_classes must match configured class order')
            if len(item.get('images',[])) != 2:
                raise ValueError('each sample requires images [BF, HAADF]')
            if tasks.nanoparticles and 'points' not in item:
                raise ValueError('nanoparticles requires points (use [] for a negative image)')
            if tasks.segmentation and not item.get('semantic_mask'):
                raise ValueError('segmentation requires semantic_mask')
        self.tasks, self.size, self.augment = tasks, tuple(size), augment
        self.max_num_centers = max_num_centers
        self.return_image = True

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = load_stem_image(*(self.root / p for p in item['images']))
        old_w, old_h = image.size
        if 'annotation_size' in item and item['annotation_size'] != [old_w, old_h]:
            raise ValueError('annotation dimensions do not match paired images')
        w,h = self.size
        image = image.resize(self.size, Image.Resampling.BILINEAR)
        sample = dict(image=torch.from_numpy(np.array(image).transpose(2,0,1).copy()).float(),
                      name=item['images'][0], im_name=item['images'][0], index=index,
                      im_size=(w,h), org_im_size=np.array([old_w,old_h]))
        if self.tasks.nanoparticles:
            from .annotations import parse_point_radius
            points = []
            for point in item['points']:
                x,y,r = parse_point_radius(point)
                if not (0 <= x < old_w and 0 <= y < old_h):
                    raise ValueError('particle center outside image bounds')
                points.append([x*w/old_w,y*h/old_h,r*(w/old_w+h/old_h)/2])
            targets = build_targets(w,h,points,support_radius=15)
            if len(points) >= self.max_num_centers:
                raise ValueError('too many particle centers')
            centers = torch.zeros(self.max_num_centers,2)
            if points:
                centers[1:len(points)+1] = torch.tensor(targets['centers'])
            label = torch.from_numpy(targets['label']).unsqueeze(0)
            sample.update(center=centers, label=label, mask=label>0,
                instance=torch.from_numpy(targets['instance']).unsqueeze(0),
                ignore=torch.zeros_like(label,dtype=torch.uint8),
                shape_coef=torch.from_numpy(targets['shape_coef']))
        if self.tasks.segmentation:
            with Image.open(self.root / item['semantic_mask']) as mask:
                if mask.size != (old_w,old_h):
                    raise ValueError('semantic mask and paired image dimensions differ')
                original = np.array(mask)
                if original.ndim != 2 or not np.issubdtype(original.dtype,np.integer):
                    raise ValueError('semantic mask must be single-channel integer class IDs')
                if not np.isin(original, [*range(len(self.tasks.classes)),255]).all():
                    raise ValueError('semantic mask contains an unknown class id')
                resized = np.array(mask.resize(self.size,Image.Resampling.NEAREST)).copy()
            sample['semantic_segmentation'] = torch.from_numpy(resized).long()
        if self.augment:
            for dim,coord,extent in [(-1,0,w),(-2,1,h)]:
                if torch.rand(()) < .5:
                    for key in ('image','label','mask','instance','ignore','shape_coef','semantic_segmentation'):
                        if key in sample:
                            sample[key] = sample[key].flip(dim)
                    if 'center' in sample and points:
                        sample['center'][1:len(points)+1,coord] = extent-1-sample['center'][1:len(points)+1,coord]
        return sample
