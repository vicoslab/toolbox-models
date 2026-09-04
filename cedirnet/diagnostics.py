"""Compact MLflow diagnostics for CeDiRNet-3DoF training."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patheffects import Normal, SimpleLineShadow


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _display_image(image):
    image = _as_numpy(image)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = image[:3].transpose(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if np.issubdtype(image.dtype, np.floating) and image.size and image.max() > 1:
        image = image / 255.0
    return np.clip(image, 0, 1) if np.issubdtype(image.dtype, np.floating) else image


def center_direction_angle_map(direction_output):
    """Convert CeDiRNet center-direction sin/cos channels to radians."""
    direction_output = _as_numpy(direction_output)
    if direction_output.ndim != 3 or direction_output.shape[0] < 2:
        raise ValueError("center-direction output must have shape CxHxW with C >= 2")
    return np.arctan2(direction_output[0], direction_output[1])


def localization_probability_map(localization_response):
    """Return a display-safe 0..1 localization response map."""
    response = np.squeeze(_as_numpy(localization_response))
    if response.ndim != 2:
        raise ValueError("localization response must reduce to a two-dimensional map")
    return np.clip(response, 0.0, 1.0)


def training_artifact_path(epoch, sample_name, subset):
    """Build a traversal-safe, epoch-scoped MLflow artifact path."""
    if subset not in {"training", "validation"}:
        raise ValueError("subset must be 'training' or 'validation'")
    normalized = str(sample_name).replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in {".", "..", "/"}]
    stem_parts = [PurePosixPath(part).stem if index == len(parts) - 1 else part for index, part in enumerate(parts)]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(stem_parts)).strip("._-") or "sample"
    return f"{subset}/epoch-{epoch + 1:04d}/{safe}-diagnostics.png"


def _draw_detections(axis, centers, scores, angles, *, distance=30):
    centers = _as_numpy(centers)
    scores = _as_numpy(scores).reshape(-1)
    angles = _as_numpy(angles).reshape(-1)
    for center, score, angle_degrees in zip(centers, scores, angles):
        x, y = center[:2]
        angle = np.deg2rad(angle_degrees)
        dx, dy = np.cos(angle) * distance, np.sin(angle) * distance
        axis.annotate("", xytext=(x, y), xy=(x + dx, y + dy), arrowprops={"color": "lime", "arrowstyle": "->"})
        axis.annotate(
            f"{score:.2f}",
            xy=(x - dx / 4, y - dy / 4),
            size="xx-small",
            ha="center",
            va="center",
            color="lime",
            path_effects=[SimpleLineShadow(shadow_color="black", linewidth=1, offset=(0, 0), alpha=0.7), Normal()],
        )


def plot_training_diagnostics(
    *,
    image,
    centers,
    scores,
    angles,
    direction_output,
    localization_response,
    ground_truth_centers=(),
):
    """Create the four views needed to inspect one augmented train sample."""
    image = _display_image(image)
    ground_truth_centers = _as_numpy(ground_truth_centers)
    if ground_truth_centers.size == 0:
        ground_truth_centers = np.empty((0, 2), dtype=np.float32)
    else:
        ground_truth_centers = ground_truth_centers.reshape(-1, ground_truth_centers.shape[-1])
        ground_truth_centers = ground_truth_centers[
            np.logical_or(ground_truth_centers[:, 0] != 0, ground_truth_centers[:, 1] != 0)
        ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].imshow(image)
    if len(ground_truth_centers):
        axes[0, 0].scatter(
            ground_truth_centers[:, 0],
            ground_truth_centers[:, 1],
            marker="+",
            s=80,
            linewidths=2,
            color="yellow",
        )
    axes[0, 0].set_title("Training sample + ground truth")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    _draw_detections(axes[0, 1], centers, scores, angles)
    axes[0, 1].set_title("Final detections")
    axes[0, 1].axis("off")

    direction = axes[1, 0].imshow(
        center_direction_angle_map(direction_output),
        cmap="jet",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1, 0].set_title("Center-direction angle")
    axes[1, 0].axis("off")
    fig.colorbar(direction, ax=axes[1, 0], fraction=0.046, pad=0.04, label="radians")

    localization = axes[1, 1].imshow(
        localization_probability_map(localization_response),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Localization probability")
    axes[1, 1].axis("off")
    fig.colorbar(localization, ax=axes[1, 1], fraction=0.046, pad=0.04, label="response")

    return fig
