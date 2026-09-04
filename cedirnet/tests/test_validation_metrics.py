import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import scipy


MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]


def cedirnet_reference_match(predictions, ground_truth):
    cost = scipy.spatial.distance_matrix(predictions, ground_truth)
    cost[cost > 20] = sys.float_info.max
    predicted_indexes, ground_truth_indexes = scipy.optimize.linear_sum_assignment(
        1.0 / (cost + 1e-10), maximize=True
    )
    valid = np.where(cost[predicted_indexes, ground_truth_indexes] < 20)[0]
    return predicted_indexes[valid], ground_truth_indexes[valid]


def load_module():
    path = MODEL_DIR / "validation_metrics.py"
    spec = importlib.util.spec_from_file_location("cedirnet_validation_metrics", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load validation_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scheduling_module():
    path = MODEL_DIR / "scheduling.py"
    spec = importlib.util.spec_from_file_location("cedirnet_scheduling", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load scheduling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidationMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_perfect_predictions_report_perfect_detection_and_zero_errors(self):
        metrics = self.module.ValidationMetrics(score_threshold=0.5, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.array([[10, 20], [40, 50]], dtype=np.float32),
            predicted_scores=np.array([0.9, 0.8], dtype=np.float32),
            predicted_angles_deg=np.array([5, 350], dtype=np.float32),
            ground_truth_centers=np.array([[10, 20], [40, 50]], dtype=np.float32),
            ground_truth_angles_deg=np.array([5, 350], dtype=np.float32),
        )
        result = metrics.compute()
        self.assertEqual(result["point_tp"], 2)
        self.assertEqual(result["point_fp"], 0)
        self.assertEqual(result["point_fn"], 0)
        self.assertEqual(result["point_precision_at_20px"], 1.0)
        self.assertEqual(result["point_recall_at_20px"], 1.0)
        self.assertEqual(result["point_f1_at_20px"], 1.0)
        self.assertEqual(result["localization_mae_px"], 0.0)
        self.assertEqual(result["orientation_mae_deg"], 0.0)

    def test_global_matching_counts_unmatched_predictions_and_ground_truth(self):
        metrics = self.module.ValidationMetrics(score_threshold=0.5, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.array([[11, 10], [80, 80]], dtype=np.float32),
            predicted_scores=np.array([0.9, 0.8], dtype=np.float32),
            predicted_angles_deg=np.array([0, 0], dtype=np.float32),
            ground_truth_centers=np.array([[10, 10], [40, 40]], dtype=np.float32),
            ground_truth_angles_deg=np.array([0, 0], dtype=np.float32),
        )
        result = metrics.compute()
        self.assertEqual((result["point_tp"], result["point_fp"], result["point_fn"]), (1, 1, 1))
        self.assertAlmostEqual(result["point_precision_at_20px"], 0.5)
        self.assertAlmostEqual(result["point_recall_at_20px"], 0.5)
        self.assertAlmostEqual(result["point_f1_at_20px"], 0.5)
        self.assertAlmostEqual(result["localization_mae_px"], 1.0)

    def test_score_threshold_is_applied_before_matching(self):
        metrics = self.module.ValidationMetrics(score_threshold=0.5, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.array([[10, 10], [20, 20]], dtype=np.float32),
            predicted_scores=np.array([0.49, 0.5], dtype=np.float32),
            predicted_angles_deg=np.array([0, 0], dtype=np.float32),
            ground_truth_centers=np.array([[10, 10], [20, 20]], dtype=np.float32),
            ground_truth_angles_deg=np.array([0, 0], dtype=np.float32),
        )
        result = metrics.compute()
        self.assertEqual((result["point_tp"], result["point_fp"], result["point_fn"]), (1, 0, 1))

    def test_f1_uses_cedirnet_strict_twenty_pixel_matching(self):
        metrics = self.module.ValidationMetrics(score_threshold=0, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.array([[19.999, 0], [100, 0]], dtype=np.float32),
            predicted_scores=np.array([1, 1], dtype=np.float32),
            predicted_angles_deg=np.array([0, 0], dtype=np.float32),
            ground_truth_centers=np.array([[0, 0], [80, 0]], dtype=np.float32),
            ground_truth_angles_deg=np.array([0, 0], dtype=np.float32),
        )
        result = metrics.compute()
        self.assertEqual((result["point_tp"], result["point_fp"], result["point_fn"]), (1, 1, 1))
        self.assertAlmostEqual(result["point_f1_at_20px"], 0.5)

    def test_matching_maximizes_valid_pairs_before_minimizing_distance(self):
        metrics = self.module.ValidationMetrics(score_threshold=0, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.array([
                [60.5056, 63.7997], [67.6450, 15.0788], [44.0313, 23.9564]
            ]),
            predicted_scores=np.ones(3),
            predicted_angles_deg=np.zeros(3),
            ground_truth_centers=np.array([
                [40.2498, 9.6704], [96.7828, 21.5004], [67.1765, 30.0420]
            ]),
            ground_truth_angles_deg=np.zeros(3),
        )
        result = metrics.compute()
        self.assertEqual((result["point_tp"], result["point_fp"], result["point_fn"]), (2, 1, 1))

    def test_orientation_error_wraps_across_zero_degrees(self):
        metrics = self.module.ValidationMetrics(score_threshold=0, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.array([[10, 10]], dtype=np.float32),
            predicted_scores=np.array([1], dtype=np.float32),
            predicted_angles_deg=np.array([359], dtype=np.float32),
            ground_truth_centers=np.array([[10, 10]], dtype=np.float32),
            ground_truth_angles_deg=np.array([1], dtype=np.float32),
        )
        result = metrics.compute()
        self.assertAlmostEqual(result["orientation_mae_deg"], 2.0)
        self.assertAlmostEqual(result["orientation_rmse_deg"], 2.0)

    def test_empty_predictions_and_ground_truth_report_undefined_errors(self):
        metrics = self.module.ValidationMetrics(score_threshold=0.5, match_centers=cedirnet_reference_match)
        metrics.update(
            predicted_centers=np.empty((0, 2)),
            predicted_scores=np.empty((0,)),
            predicted_angles_deg=np.empty((0,)),
            ground_truth_centers=np.empty((0, 2)),
            ground_truth_angles_deg=np.empty((0,)),
        )
        result = metrics.compute()
        self.assertEqual(result["matched_points"], 0)
        self.assertTrue(np.isnan(result["localization_mae_px"]))
        self.assertTrue(np.isnan(result["orientation_mae_deg"]))

    def test_ground_truth_angles_are_sampled_at_xy_centers(self):
        orientation = np.zeros((1, 8, 9), dtype=np.float32)
        orientation[0, 3, 5] = np.pi / 2
        centers, angles = self.module.extract_ground_truth(
            np.array([[5, 3], [0, 0]], dtype=np.float32), orientation
        )
        np.testing.assert_array_equal(centers, [[5, 3]])
        np.testing.assert_allclose(angles, [90.0], atol=1e-6)

    def test_ground_truth_rejects_invalid_sentinels_but_keeps_edge_centers(self):
        orientation = np.zeros((1, 8, 9), dtype=np.float32)
        centers, angles = self.module.extract_ground_truth(
            np.array([[-1, -1], [0, 5], [8, 7], [9, 1], [0, 0]], dtype=np.float32),
            orientation,
        )
        np.testing.assert_array_equal(centers, [[0, 5], [8, 7]])
        np.testing.assert_array_equal(angles, [0, 0])


class ValidationIntegrationContractTest(unittest.TestCase):
    def test_validation_processes_all_images_and_logs_mlflow_metrics(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("def validate(self, epoch):", train)
        self.assertIn("ValidationMetrics(", train)
        self.assertIn("mlflow.log_metrics(validation_metrics, step=epoch)", train)
        self.assertIn("self.visualize_sample(", train)
        self.assertIn("subset='validation'", train)
        self.assertNotIn("self.visualize(self.validation_dataset_it", train)
        self.assertIn("should_validate(", train)

    def test_train_visualization_remains_limited(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("self.visualize_training_samples(epoch)", train)
        self.assertIn("visualized >= self.args['visualization_samples']", train)
        self.assertIn("self.training_visualization_dataset_it", train)
        self.assertIn("num_workers=0", train)
        self.assertNotIn("for sample in tqdm(self.train_dataset_it, desc='visualise training'", train)

    def test_five_hundred_epoch_run_schedules_all_ten_validations(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("from scheduling import should_validate", train)
        scheduling = load_scheduling_module()
        triggered = [
            epoch + 1 for epoch in range(500)
            if scheduling.should_validate(epoch, 500, 50)
        ]
        self.assertEqual(triggered, list(range(50, 501, 50)))

    def test_validation_metric_options_are_exposed(self):
        schema = __import__("json").loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["validation_score_threshold"]["default"], 0.5)
        self.assertNotIn("validation_match_distance", schema["properties"])
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("CenterGlobalMinimizationEval(tau_thr=POINT_MATCH_DISTANCE_PX)", train)
        self.assertIn("mlflow.log_param('validation_match_distance_px', POINT_MATCH_DISTANCE_PX)", train)
        self.assertIn("validation/point_f1_at_20px", train)


if __name__ == "__main__":
    unittest.main()
