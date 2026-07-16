"""Point-and-radius annotation helpers for the CeDiRNet-STEM adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


def infer_haadf_path(bf_path: str | Path) -> Path:
    """Resolve the conventional ``*_HAADF`` sibling of a ``*_BF`` image."""
    bf_path = Path(bf_path)
    marker = "_BF."
    if marker not in bf_path.name:
        raise ValueError(
            "paired STEM image path must contain '_BF.' or '_HAADF.'"
        )
    return bf_path.with_name(bf_path.name.replace(marker, "_HAADF.", 1))


def infer_bf_path(haadf_path: str | Path) -> Path:
    """Resolve the conventional ``*_BF`` sibling of a ``*_HAADF`` image."""
    haadf_path = Path(haadf_path)
    marker = "_HAADF."
    if marker not in haadf_path.name:
        raise ValueError(
            "paired STEM image path must contain '_BF.' or '_HAADF.'"
        )
    return haadf_path.with_name(haadf_path.name.replace(marker, "_BF.", 1))


def load_stem_image(bf_source, haadf_source=None) -> Image.Image:
    """Compose the upstream STEM input as BF, HAADF, and a zero channel."""
    if haadf_source is None:
        if not isinstance(bf_source, (str, Path)):
            raise ValueError("HAADF source is required for uploaded BF images")
        primary_source = Path(bf_source)
        if "_HAADF." in primary_source.name:
            haadf_source = primary_source
            bf_source = infer_bf_path(primary_source)
        else:
            haadf_source = infer_haadf_path(primary_source)

    for channel_name, source in (("BF", bf_source), ("HAADF", haadf_source)):
        if isinstance(source, (str, Path)) and not Path(source).is_file():
            raise FileNotFoundError(f"{channel_name} image does not exist: {source}")

    with Image.open(bf_source) as bf_image:
        bf = np.asarray(ImageOps.exif_transpose(bf_image).convert("L"))
    with Image.open(haadf_source) as haadf_image:
        haadf = np.asarray(ImageOps.exif_transpose(haadf_image).convert("L"))

    if bf.shape != haadf.shape:
        raise ValueError(
            f"BF and HAADF image dimensions differ: {bf.shape} != {haadf.shape}"
        )
    channels = np.stack((bf, haadf, np.zeros_like(bf)), axis=2)
    return Image.fromarray(channels.astype(np.uint8))


def parse_point_radius(annotation: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(center_x, center_y, radius)`` from a supported annotation.

    The canonical manifest form is ``[x, y, radius]``.  Toolbox Vector
    annotations are exported as ``[x, y, radius_x, radius_y]``; in that form
    the Euclidean distance between the two vertices is used as the radius.
    """
    if len(annotation) == 3:
        x, y, radius = map(float, annotation)
    elif len(annotation) == 4:
        x, y, radius_x, radius_y = map(float, annotation)
        radius = math.hypot(radius_x - x, radius_y - y)
    else:
        raise ValueError(
            "point-radius annotation must be [x, y, radius] or "
            "[x, y, radius_x, radius_y]"
        )

    if not all(math.isfinite(value) for value in (x, y, radius)):
        raise ValueError("point-radius annotation values must be finite")
    if radius <= 0:
        raise ValueError("radius must be positive")
    return x, y, radius


def build_targets(
    width: int,
    height: int,
    annotations: Iterable[Sequence[float]],
    *,
    support_radius: int = 15,
) -> dict[str, object]:
    """Build CeDiRNet center, instance, label, and radius target arrays."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if support_radius < 1:
        raise ValueError("support_radius must be at least one pixel")

    label = np.zeros((height, width), dtype=np.uint8)
    instance = np.zeros((height, width), dtype=np.int16)
    shape_coef = np.zeros((1, height, width), dtype=np.float32)
    centers: list[tuple[float, float]] = []

    for instance_id, raw_annotation in enumerate(annotations, start=1):
        x, y, radius = parse_point_radius(raw_annotation)
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError(
                f"annotation center ({x}, {y}) is outside image bounds "
                f"{width}x{height}"
            )

        center_x, center_y = int(round(x)), int(round(y))
        center_x = min(max(center_x, 0), width - 1)
        center_y = min(max(center_y, 0), height - 1)
        x0 = max(0, center_x - support_radius)
        x1 = min(width, center_x + support_radius + 1)
        y0 = max(0, center_y - support_radius)
        y1 = min(height, center_y + support_radius + 1)

        label[y0:y1, x0:x1] = 1
        instance[y0:y1, x0:x1] = instance_id
        shape_coef[0, y0:y1, x0:x1] = radius
        centers.append((x, y))

    return {
        "centers": centers,
        "label": label,
        "instance": instance,
        "shape_coef": shape_coef,
    }
