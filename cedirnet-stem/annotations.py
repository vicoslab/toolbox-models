"""Point-and-radius annotation helpers for the CeDiRNet-STEM adapter."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


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
