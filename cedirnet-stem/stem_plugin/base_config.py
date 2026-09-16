"""CeDiRNet-STEM configuration for toolbox point+radius manifests."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import InterpolationMode

from utils import transforms as my_transforms


NUM_VECTOR_FIELDS = 4  # sin, cos, center distance, particle radius
MAX_NUM_CENTERS = 2048


def get_args(width=512, height=512, batch_size=2, workers=2):
    transform = my_transforms.get_transform(
        [
            {
                "name": "ToTensor",
                "opts": {
                    "keys": (
                        "image",
                        "instance",
                        "label",
                        "ignore",
                        "mask",
                        "shape_coef",
                    ),
                    "type": (
                        torch.FloatTensor,
                        torch.ShortTensor,
                        torch.ByteTensor,
                        torch.ByteTensor,
                        torch.ByteTensor,
                        torch.FloatTensor,
                    ),
                },
            },
            {
                "name": "Resize",
                "opts": {
                    "keys": (
                        "image",
                        "instance",
                        "label",
                        "ignore",
                        "mask",
                        "shape_coef",
                    ),
                    "interpolation": (
                        InterpolationMode.BILINEAR,
                        InterpolationMode.NEAREST,
                        InterpolationMode.NEAREST,
                        InterpolationMode.NEAREST,
                        InterpolationMode.NEAREST,
                        InterpolationMode.NEAREST,
                    ),
                    "keys_bbox": ("center",),
                    "keys_custom_fn": {
                        "shape_coef": lambda values, old_size, new_size: values
                        * float(np.mean(np.asarray(new_size) / np.asarray(old_size)))
                    },
                    "size": (height, width),
                },
            },
            {
                "name": "RandomHorizontalFlip",
                "opts": {
                    "keys": ("image", "instance", "label", "ignore", "mask", "shape_coef"),
                    "keys_bbox": ("center",),
                    "p": 0.5,
                },
            },
            {
                "name": "RandomVerticalFlip",
                "opts": {
                    "keys": ("image", "instance", "label", "ignore", "mask", "shape_coef"),
                    "keys_bbox": ("center",),
                    "p": 0.5,
                },
            },
            {
                "name": "ColorJitter",
                "opts": {
                    "keys": ("image",),
                    "p": 0.5,
                    "saturation": 0.3,
                    "hue": 0.3,
                    "brightness": 0.3,
                    "contrast": 0.3,
                },
            },
        ]
    )

    args = {
        "cuda": torch.cuda.is_available(),
        "display": False,
        "save": True,
        "n_epochs": 100,
        "save_interval": 10,
        "pretrained_model_path": None,
        "pretrained_center_model_path": None,
        "train_dataset": {
            "name": "generic_point_radius",
            "kwargs": {
                "manifest": None,
                "split": "train",
                "fixed_bbox_size": 15,
                "max_num_centers": MAX_NUM_CENTERS,
                "transform": transform,
            },
            "centerdir_gt_opts": {
                "generic_regression_maps": [("shape_coef", "gt_shape_coef")],
                "ignore_instance_mask_and_use_closest_center": True,
                "center_ignore_px": 3,
                "skip_gt_center_mask_generate": True,
                "MAX_NUM_CENTERS": MAX_NUM_CENTERS,
            },
            "batch_size": batch_size,
            "hard_samples_size": 0,
            "workers": workers,
            "shuffle": True,
        },
        "model": {
            "name": "fpn",
            "kwargs": {
                "backbone": "tu-convnext_base",
                "num_classes": [NUM_VECTOR_FIELDS, 1],
                "use_custom_fpn": True,
                "add_output_exp": False,
                "in_channels": 3,
                "fpn_args": {
                    "upsampling": 4,
                    "decoder_segmentation_head_channels": 64,
                    "classes_grouping": [(0, 1, 2, 4), (3,)],
                },
                "init_decoder_gain": 0.1,
            },
            "optimizer": "Adam",
            "lr": 1e-4,
            "weight_decay": 0,
        },
        "center_model": {
            "name": "CenterAttributeEstimator",
            "use_learnable_center_estimation": True,
            "kwargs": {
                "attributes": {
                    "shape_coef": {
                        "input_start": 0,
                        "input_end": 1,
                        "use_log": True,
                        "use_log_channels": [0],
                    }
                },
                "use_centerdir_radii": False,
                "use_magnitude_as_mask": True,
                "local_max_thr": 0.01,
                "ignore_centerdir_magnitude": True,
                "ignore_cls_prediction": True,
                "use_dilated_nn": True,
                "dilated_nn_args": {
                    "return_sigmoid": False,
                    "inner_ch": 16,
                    "inner_kernel": 3,
                    "dilations": [1, 4, 8, 12],
                    "use_centerdir_radii": False,
                    "use_centerdir_magnitude": False,
                    "use_cls_mask": False,
                },
                "augmentation": False,
            },
            "optimizer": "Adam",
            "lr": 0,
            "weight_decay": 0,
        },
        "loss_type": "ShapeLoss",
        "loss_opts": {
            "num_vector_fields": NUM_VECTOR_FIELDS,
            "foreground_weight": 1,
            "no_instance_loss": True,
            "cls_no_loss": True,
            "cls_instance_weighted": True,
            "centerdir_instance_weighted": True,
            "cls_loss": "l1",
            "regression_loss": "l1",
            "learnable_center_est": False,
            "enable_centerdir_loss": True,
            "enable_cls_loss": False,
            "shape_args": {
                "enable": True,
                "shape_type": "circle",
                "no_instance_loss": False,
                "regression_loss": "l1",
                "individual_uw": False,
                "use_log": True,
                "use_log_channels": [0],
            },
        },
        "loss_w": {
            "w_cent": 0.1,
            "w_shape": 0.5,
            "w_radius": 1.0,
        },
    }

    args["model"]["lambda_scheduler_fn"] = lambda all_args: (
        lambda epoch: pow(1 - epoch / all_args["n_epochs"], 0.9)
    )
    args["center_model"]["lambda_scheduler_fn"] = lambda _all_args: (
        lambda _epoch: 1.0
    )
    return args
