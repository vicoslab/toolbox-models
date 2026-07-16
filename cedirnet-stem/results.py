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


def label_studio_vector_result(
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
    """Encode center+radius as a Label Studio two-vertex Vector result."""
    center_x, center_y = map(float, center)
    original_width, original_height = map(float, original_size)
    x_percent = center_x * 100.0
    y_percent = center_y * 100.0
    radius = float(radius)
    candidates = [
        (x_percent + radius / original_width * 100.0, y_percent),
        (x_percent - radius / original_width * 100.0, y_percent),
        (x_percent, y_percent + radius / original_height * 100.0),
        (x_percent, y_percent - radius / original_height * 100.0),
    ]
    handle_x, handle_y = next(
        (
            candidate
            for candidate in candidates
            if 0.0 <= candidate[0] <= 100.0 and 0.0 <= candidate[1] <= 100.0
        ),
        candidates[0],
    )

    return {
        "id": result_id,
        "from_name": from_name,
        "to_name": to_name,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "image_rotation": 0,
        "value": {
            "closed": False,
            "vertices": [
                {"x": x_percent, "y": y_percent, "id": f"{result_id}-center"},
                {
                    "x": handle_x,
                    "y": handle_y,
                    "id": f"{result_id}-radius",
                },
            ],
            "labels": [label],
        },
        "score": float(score),
        "type": "labels",
        "readonly": False,
    }
