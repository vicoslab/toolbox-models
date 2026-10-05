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


class BestF1Metrics:
    """Select best F1 from a single sorted split-wide precision–recall curve.

    Match all candidates once per image using CeDiRNet's spatial assignment.
    Keep those TP/FP labels fixed, sort by score, and accumulate prefix counts.
    Equal scores enter together so the selected point is realizable by a score
    threshold. Equal F1 prefers the highest threshold, including no detections.
    """

    def __init__(self, *, match_centers):
        self.match_centers = match_centers
        self.records = []
        self.images = 0
        self.ground_truth_points = 0

    def update(self, *, predicted_centers, predicted_scores, predicted_angles_deg,
               ground_truth_centers, ground_truth_angles_deg):
        centers = np.asarray(predicted_centers, dtype=np.float64).reshape(-1, 2)
        scores = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
        angles = np.asarray(predicted_angles_deg, dtype=np.float64).reshape(-1)
        targets = np.asarray(ground_truth_centers, dtype=np.float64).reshape(-1, 2)
        target_angles = np.asarray(ground_truth_angles_deg, dtype=np.float64).reshape(-1)
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('candidate scores must be finite and non-negative')
        if not (len(centers) == len(scores) == len(angles)):
            raise ValueError('prediction centers, scores, and angles must have equal length')
        if len(targets) != len(target_angles):
            raise ValueError('ground-truth centers and angles must have equal length')
        # Each row stores score, fixed TP flag, localization error, angle error.
        records = np.zeros((len(scores), 4), dtype=np.float64)
        records[:, 0] = scores
        if len(centers) and len(targets):
            rows, cols = self.match_centers(centers, targets)
            rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
            records[rows, 1] = 1
            records[rows, 2] = np.linalg.norm(centers[rows] - targets[cols], axis=1)
            records[rows, 3] = np.abs((angles[rows] - target_angles[cols] + 180) % 360 - 180)
        self.records.append(records)
        self.images += 1
        self.ground_truth_points += len(targets)

    def compute(self):
        records = np.concatenate(self.records) if self.records else np.empty((0, 4))
        candidate_points = len(records)
        if candidate_points:
            records = records[np.argsort(-records[:, 0], kind='stable')]
            scores = records[:, 0]
            # Curve points at the end of each tied-score group. All prefixes are
            # accumulated once; no threshold loop, matching, or model replay.
            ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), candidate_points - 1]
            prefix_tp = np.cumsum(records[:, 1], dtype=np.int64)
            counts = ends + 1
            tp = prefix_tp[ends]
            f1 = 2.0 * tp / (counts + self.ground_truth_points)
            winner = int(np.argmax(f1))
            if f1[winner] > 0:
                count = int(counts[winner])
                threshold = float(scores[count - 1])
            else:
                count = 0
                threshold = float(np.nextafter(np.float32(scores[0]), np.float32(np.inf)))
                if not np.isfinite(threshold) or threshold <= scores[0]:
                    threshold = float(np.nextafter(scores[0], np.inf))
        else:
            count, threshold = 0, 0.0
        selected = records[:count]
        matched = selected[selected[:, 1] == 1]
        # Reuse metric formatting, not its matcher. Errors/counts correspond to
        # the same fixed-label PR point that supplied the chosen score.
        metrics = ValidationMetrics(score_threshold=threshold, match_centers=self.match_centers)
        metrics.tp = len(matched)
        metrics.fp = count - metrics.tp
        metrics.fn = self.ground_truth_points - metrics.tp
        metrics.images = self.images
        metrics.predicted_points = count
        metrics.ground_truth_points = self.ground_truth_points
        metrics.localization_errors = matched[:, 2].tolist()
        metrics.orientation_errors = matched[:, 3].tolist()
        result = metrics.compute()
        result['best_f1_score_threshold'] = threshold
        result['candidate_points'] = float(candidate_points)
        return result


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
