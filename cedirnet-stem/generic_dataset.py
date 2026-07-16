"""Manifest dataset adapter for point+radius CeDiRNet-STEM annotations."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .annotations import build_targets, load_stem_image
from utils import transforms as my_transforms


class GenericPointRadiusDataset(Dataset):
    """Load toolbox manifests containing center and particle-radius annotations."""

    DATASET_NAME = "generic_point_radius"

    def __init__(
        self,
        manifest,
        split="train",
        fixed_bbox_size=15,
        max_num_centers=2048,
        transform=None,
        num_cpu_threads=1,
        **_,
    ):
        if not manifest:
            raise ValueError("dataset is missing a manifest file")
        if num_cpu_threads:
            torch.set_num_threads(num_cpu_threads)

        with open(manifest, encoding="utf-8") as stream:
            manifest_data = json.load(stream)
        if split in manifest_data:
            items = manifest_data[split]
        elif split == "train" and "data" in manifest_data:
            items = manifest_data["data"]
        else:
            items = []

        self.root_dir = os.path.dirname(os.path.abspath(manifest))
        self.items = [item for item in items if "points" in item]
        skipped = len(items) - len(self.items)
        if skipped:
            print(f"Warning: skipped {skipped} items without point-radius annotations")
        if not self.items:
            raise ValueError(f"manifest split {split!r} contains no annotated images")

        self.fixed_bbox_size = int(fixed_bbox_size)
        self.max_num_centers = int(max_num_centers)
        self.transform = (
            my_transforms.get_transform(transform) if isinstance(transform, list) else transform
        )
        self.return_image = True

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        relative_path = item["image_path"]
        image_path = os.path.abspath(os.path.join(self.root_dir, relative_path))
        haadf_relative_path = item.get("haadf_image_path")
        haadf_path = (
            os.path.abspath(os.path.join(self.root_dir, haadf_relative_path))
            if haadf_relative_path
            else None
        )
        image = load_stem_image(image_path, haadf_path)
        width, height = image.size

        targets = build_targets(
            width,
            height,
            item["points"],
            support_radius=self.fixed_bbox_size,
        )
        if len(targets["centers"]) >= self.max_num_centers:
            raise ValueError(
                f"image {relative_path!r} contains too many centers: "
                f"{len(targets['centers'])} >= {self.max_num_centers}"
            )

        centers = np.zeros((self.max_num_centers, 2), dtype=np.float32)
        if targets["centers"]:
            centers[1 : len(targets["centers"]) + 1] = np.asarray(
                targets["centers"], dtype=np.float32
            )

        label = torch.from_numpy(targets["label"]).unsqueeze(0)
        sample = {
            "image": image,
            "im_name": relative_path,
            "name": relative_path,
            "org_im_size": np.asarray((width, height)),
            "im_size": (width, height),
            "index": index,
            "center": centers,
            "label": label,
            "mask": label > 0,
            "ignore": torch.zeros_like(label, dtype=torch.uint8),
            "instance": torch.from_numpy(targets["instance"]).unsqueeze(0),
            "shape_coef": torch.from_numpy(targets["shape_coef"]),
        }

        if self.transform is not None:
            sample = self.transform(sample, np.random.default_rng(1337 + index))
        return sample
