"""Framework-independent CeDiRNet-STEM inference result conversion."""

from __future__ import annotations

from typing import Sequence


def restore_prediction(
    center: Sequence[float],
    radius: float,
    network_size: Sequence[int],
    original_size: Sequence[int],
) -> tuple[tuple[float, float], float]:
    """Restore a resized prediction to normalized center and original-pixel radius."""
    network_width, network_height = map(float, network_size)
    original_width, original_height = map(float, original_size)
    if min(network_width, network_height, original_width, original_height) <= 0:
        raise ValueError("image dimensions must be positive")

    center_x, center_y = map(float, center)
    normalized_center = (center_x / network_width, center_y / network_height)
    resize_scale = 0.5 * (
        network_width / original_width + network_height / original_height
    )
    return normalized_center, float(radius) / resize_scale


def label_studio_ellipse_result(
    *,
    center: Sequence[float],
    radius: float,
    score: float,
    original_size: Sequence[int],
    from_name: str,
    to_name: str,
    label: str,
    result_id: str,
) -> dict[str, object]:
    """Encode the circle prediction as an ellipse with equal pixel semi-axes.

    Label Studio x/y are the ellipse center. Percentage radii differ on a
    non-square image; neither the center nor radius is clipped at image edges.
    The circle runtime does not predict eccentricity or orientation.
    """
    center_x, center_y = map(float, center)
    original_width, original_height = map(float, original_size)
    return {
        "id": result_id,
        "from_name": from_name,
        "to_name": to_name,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "image_rotation": 0,
        "value": {
            "x": center_x * 100,
            "y": center_y * 100,
            "radiusX": float(radius) / original_width * 100,
            "radiusY": float(radius) / original_height * 100,
            "rotation": 0,
            "ellipselabels": [label],
        },
        "score": float(score),
        "type": "ellipselabels",
        "readonly": False,
    }
