"""Best F1 from a sorted precision–recall curve with fixed match labels."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import scipy.optimize
import scipy.spatial

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('best_f1_metrics', ROOT / 'validation_metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def match(predictions, targets):
    distances = scipy.spatial.distance_matrix(predictions, targets)
    distances[distances > 20] = np.finfo(float).max
    rows, cols = scipy.optimize.linear_sum_assignment(1 / (distances + 1e-10), maximize=True)
    valid = distances[rows, cols] < 20
    return rows[valid], cols[valid]


def sample(points, scores, targets):
    return dict(predicted_centers=np.asarray(points).reshape(-1, 2),
                predicted_scores=np.asarray(scores), predicted_angles_deg=np.zeros(len(scores)),
                ground_truth_centers=np.asarray(targets).reshape(-1, 2),
                ground_truth_angles_deg=np.zeros(len(targets)))


class BestF1Test(unittest.TestCase):
    def best(self, samples):
        module = load()
        obj = module.BestF1Metrics(match_centers=match)
        for values in samples:
            obj.update(**values)
        return obj.compute()

    def test_low_confidence_true_prediction_is_not_prefiltered(self):
        result = self.best([sample([[10, 10], [80, 80]], [.03, .01], [[10, 10]])])
        self.assertEqual(result['best_f1_score_threshold'], .03)
        self.assertEqual(result['point_f1_at_20px'], 1)
        self.assertEqual(result['point_precision_at_20px'], 1)
        self.assertEqual(result['point_recall_at_20px'], 1)
        self.assertEqual(result['candidate_points'], 2)

    def test_threshold_is_global_for_split_not_per_image(self):
        result = self.best([sample([[10, 10]], [.9], [[10, 10]]),
                            sample([[10, 10], [80, 80]], [.1, .8], [[10, 10]])])
        self.assertEqual(result['best_f1_score_threshold'], .1)
        self.assertAlmostEqual(result['point_f1_at_20px'], .8)
        self.assertAlmostEqual(result['point_precision_at_20px'], 2/3)
        self.assertEqual(result['point_recall_at_20px'], 1)
        self.assertEqual(result['images'], 2)

    def test_equal_scores_enter_together(self):
        result = self.best([sample([[10, 10], [80, 80]], [.2, .2], [[10, 10]])])
        self.assertEqual(result['point_tp'], 1)
        self.assertEqual(result['point_fp'], 1)
        self.assertAlmostEqual(result['point_f1_at_20px'], 2/3)

    def test_full_set_assignment_labels_are_fixed_during_pr_calculation(self):
        # Upstream global distance matching selects the exact low-score point.
        # Sorting must not relabel its high-score duplicate as a true positive.
        result = self.best([sample([[10, 11], [10, 10]], [.9, .1], [[10, 10]])])
        self.assertEqual(result['best_f1_score_threshold'], .1)
        self.assertAlmostEqual(result['point_f1_at_20px'], 2/3)
        self.assertEqual(result['point_tp'], 1)
        self.assertEqual(result['point_fp'], 1)
        self.assertEqual(result['localization_mae_px'], 0)

    def test_tied_best_f1_prefers_highest_threshold(self):
        result = self.best([sample([[10, 10], [80, 80], [150, 150], [40, 40]],
                                   [.9, .8, .75, .7], [[10, 10], [40, 40]])])
        self.assertEqual(result['best_f1_score_threshold'], .9)
        self.assertEqual(result['point_tp'], 1)
        self.assertAlmostEqual(result['point_f1_at_20px'], 2/3)

    def test_negative_only_split_selects_empty_prediction_operating_point(self):
        result = self.best([sample([[10, 10]], [.3], [])])
        self.assertGreater(result['best_f1_score_threshold'], .3)
        self.assertEqual(result['point_fp'], 0)
        self.assertEqual(result['images'], 1)
        self.assertEqual(result['candidate_points'], 1)

    def test_no_candidates_has_finite_deterministic_threshold(self):
        result = self.best([sample([], [], [[10, 10]]), sample([], [], [])])
        self.assertEqual(result['best_f1_score_threshold'], 0)
        self.assertEqual(result['point_fn'], 1)
        self.assertEqual(result['point_f1_at_20px'], 0)
        self.assertEqual(result['images'], 2)

    def test_random_curves_match_fixed_labels_and_direct_prefix_counts(self):
        rng = np.random.default_rng(7)
        for trial in range(15):
            samples = [sample(rng.uniform(0, 70, (5, 2)), rng.choice([.02, .2, .7], 5),
                              rng.uniform(0, 70, (3, 2))) for _ in range(3)]
            scores, labels = [], []
            for values in samples:
                rows, _ = match(values['predicted_centers'], values['ground_truth_centers'])
                truth = np.zeros(len(values['predicted_scores']), dtype=bool)
                truth[rows] = True
                scores.extend(values['predicted_scores'])
                labels.extend(truth)
            scores, labels = np.asarray(scores), np.asarray(labels)
            thresholds = sorted(set(scores), reverse=True)
            thresholds.insert(0, float(np.nextafter(np.float32(max(thresholds)), np.float32(np.inf))))
            candidates = []
            for threshold in thresholds:
                selected = scores >= threshold
                tp = int(labels[selected].sum())
                count = int(selected.sum())
                candidates.append((2*tp/(count+9), threshold, tp, count-tp, 9-tp))
            expected = max(candidates, key=lambda x: x[0])
            actual = self.best(samples)
            with self.subTest(trial=trial):
                self.assertEqual(actual['best_f1_score_threshold'], expected[1])
                self.assertAlmostEqual(actual['point_f1_at_20px'], expected[0])
                self.assertEqual((actual['point_tp'], actual['point_fp'], actual['point_fn']), expected[2:])

    def test_dense_curve_matches_once_and_never_matches_again_in_compute(self):
        from unittest.mock import Mock, patch
        module = load()
        callback = Mock(side_effect=match)
        values = sample(np.column_stack([np.arange(2000), np.ones(2000)]),
                        np.linspace(.01, .99, 2000), [[10, 1], [70, 1]])
        with patch.object(scipy.spatial, 'distance_matrix', wraps=scipy.spatial.distance_matrix) as distances:
            obj = module.BestF1Metrics(match_centers=callback)
            obj.update(**values)
            self.assertEqual(callback.call_count, 1)
            callback.side_effect = AssertionError('no rematching at PR points or final threshold')
            obj.compute()
            obj.compute()
        self.assertEqual(callback.call_count, 1)
        self.assertEqual(distances.call_count, 1)

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.best([sample([[1, 1]], [np.nan], [[1, 1]])])


if __name__ == '__main__':
    unittest.main()
