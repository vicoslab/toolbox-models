"""Dataset-level point and orientation metrics for CeDiRNet validation."""

from __future__ import annotations

import numpy as np

POINT_MATCH_DISTANCE_PX = 20


class ValidationMetrics:
    """Accumulate globally matched point-localization and orientation metrics."""

    def __init__(self, *, score_threshold: float, match_centers):
        if not 0 <= score_threshold:
            raise ValueError("score_threshold must be non-negative")
        self.score_threshold = float(score_threshold)
        self.match_centers = match_centers
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.images = 0
        self.predicted_points = 0
        self.ground_truth_points = 0
        self.localization_errors = []
        self.orientation_errors = []

    def update(
        self,
        *,
        predicted_centers,
        predicted_scores,
        predicted_angles_deg,
        ground_truth_centers,
        ground_truth_angles_deg,
    ):
        predicted_centers = np.asarray(predicted_centers, dtype=np.float64).reshape(-1, 2)
        predicted_scores = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
        predicted_angles_deg = np.asarray(predicted_angles_deg, dtype=np.float64).reshape(-1)
        ground_truth_centers = np.asarray(ground_truth_centers, dtype=np.float64).reshape(-1, 2)
        ground_truth_angles_deg = np.asarray(ground_truth_angles_deg, dtype=np.float64).reshape(-1)

        if not (len(predicted_centers) == len(predicted_scores) == len(predicted_angles_deg)):
            raise ValueError("prediction centers, scores, and angles must have equal length")
        if len(ground_truth_centers) != len(ground_truth_angles_deg):
            raise ValueError("ground-truth centers and angles must have equal length")

        selected = predicted_scores >= self.score_threshold
        predicted_centers = predicted_centers[selected]
        predicted_angles_deg = predicted_angles_deg[selected]
        self.images += 1
        self.predicted_points += len(predicted_centers)
        self.ground_truth_points += len(ground_truth_centers)

        if len(predicted_centers) == 0 or len(ground_truth_centers) == 0:
            self.fp += len(predicted_centers)
            self.fn += len(ground_truth_centers)
            return

        predicted_indexes, ground_truth_indexes = self.match_centers(
            predicted_centers, ground_truth_centers
        )
        predicted_indexes = np.asarray(predicted_indexes, dtype=int)
        ground_truth_indexes = np.asarray(ground_truth_indexes, dtype=int)
        matched_distances = np.linalg.norm(
            predicted_centers[predicted_indexes] - ground_truth_centers[ground_truth_indexes],
            axis=1,
        )

        matched = len(predicted_indexes)
        self.tp += matched
        self.fp += len(predicted_centers) - matched
        self.fn += len(ground_truth_centers) - matched
        self.localization_errors.extend(matched_distances.tolist())

        angle_delta = (
            predicted_angles_deg[predicted_indexes]
            - ground_truth_angles_deg[ground_truth_indexes]
            + 180.0
        ) % 360.0 - 180.0
        self.orientation_errors.extend(np.abs(angle_delta).tolist())

    def compute(self):
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        localization = np.asarray(self.localization_errors, dtype=np.float64)
        orientation = np.asarray(self.orientation_errors, dtype=np.float64)

        return {
            "point_precision_at_20px": float(precision),
            "point_recall_at_20px": float(recall),
            "point_f1_at_20px": float(f1),
            "point_tp": float(self.tp),
            "point_fp": float(self.fp),
            "point_fn": float(self.fn),
            "localization_mae_px": float(localization.mean()) if localization.size else float("nan"),
            "localization_rmse_px": float(np.sqrt(np.mean(localization ** 2))) if localization.size else float("nan"),
            "orientation_mae_deg": float(orientation.mean()) if orientation.size else float("nan"),
            "orientation_rmse_deg": float(np.sqrt(np.mean(orientation ** 2))) if orientation.size else float("nan"),
            "matched_points": float(len(localization)),
            "predicted_points": float(self.predicted_points),
            "ground_truth_points": float(self.ground_truth_points),
            "images": float(self.images),
        }


def extract_ground_truth(centers, orientation_map):
    """Extract valid XY centers and orientation degrees from a sample map."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    orientation_map = np.squeeze(np.asarray(orientation_map, dtype=np.float64))
    if orientation_map.ndim != 2:
        raise ValueError("orientation map must reduce to HxW")

    height, width = orientation_map.shape
    not_padding = np.logical_or(centers[:, 0] != 0, centers[:, 1] != 0)
    in_bounds = (
        (centers[:, 0] >= 0)
        & (centers[:, 1] >= 0)
        & (centers[:, 0] < width)
        & (centers[:, 1] < height)
    )
    centers = centers[np.logical_and(not_padding, in_bounds)]
    if len(centers) == 0:
        return centers, np.empty((0,), dtype=np.float64)

    x = centers[:, 0].astype(int)
    y = centers[:, 1].astype(int)
    angles = np.rad2deg(orientation_map[y, x]) % 360.0
    return centers, angles
