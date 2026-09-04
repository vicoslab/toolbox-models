import importlib.util
import pathlib
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np


MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_module():
    path = MODEL_DIR / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("cedirnet_diagnostics", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiagnosticMapsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_center_direction_map_uses_atan2_sin_cos(self):
        output = np.zeros((5, 2, 2), dtype=np.float32)
        output[0] = [[0, 1], [0, -1]]
        output[1] = [[1, 0], [-1, 0]]
        actual = self.module.center_direction_angle_map(output)
        expected = np.array([[0, np.pi / 2], [np.pi, -np.pi / 2]])
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_localization_probability_is_clipped_to_unit_interval(self):
        response = np.array([[-2.0, 0.25], [0.8, 3.0]])
        actual = self.module.localization_probability_map(response)
        np.testing.assert_allclose(actual, [[0.0, 0.25], [0.8, 1.0]])

    def test_artifact_path_is_nested_and_sanitized(self):
        actual = self.module.training_artifact_path(3, "../board/tile.jpg", "training")
        self.assertEqual(actual, "visualizations/epoch_0004/training/board_tile-diagnostics.png")

    def test_validation_artifact_path_uses_separate_subfolder(self):
        actual = self.module.training_artifact_path(3, "../board/tile.jpg", "validation")
        self.assertEqual(actual, "visualizations/epoch_0004/validation/board_tile-diagnostics.png")

    def test_artifact_path_rejects_unknown_subset(self):
        with self.assertRaisesRegex(ValueError, "subset"):
            self.module.training_artifact_path(3, "tile.jpg", "testing")

    def test_diagnostic_figure_contains_four_named_panels_and_jet_direction_map(self):
        image = np.zeros((16, 24, 3), dtype=np.uint8)
        output = np.zeros((5, 16, 24), dtype=np.float32)
        heatmap = np.full((16, 24), 0.5, dtype=np.float32)
        fig = self.module.plot_training_diagnostics(
            image=image,
            centers=np.array([[8, 6, 1]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            angles=np.array([0.0], dtype=np.float32),
            direction_output=output,
            localization_response=heatmap,
            ground_truth_centers=np.array([[5, 4]], dtype=np.float32),
        )
        titles = [axis.get_title() for axis in fig.axes]
        self.assertIn("Training sample + ground truth", titles)
        self.assertIn("Final detections", titles)
        self.assertIn("Center-direction angle", titles)
        self.assertIn("Localization probability", titles)
        direction_axis = next(axis for axis in fig.axes if axis.get_title() == "Center-direction angle")
        self.assertEqual(direction_axis.images[0].get_cmap().name, "jet")
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "diagnostics.png"
            fig.savefig(target)
            self.assertGreater(target.stat().st_size, 0)


class PreparedModelContractTest(unittest.TestCase):
    def test_setup_downloads_default_localization_checkpoint(self):
        setup = (MODEL_DIR / "setup.sh").read_text(encoding="utf-8")
        self.assertIn("localization_checkpoint.pth", setup)
        self.assertIn("https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth", setup)
        self.assertIn("cffcfde184a22c03a67ecc741f3943d0325d4aabe812cb1787796f236403df84", setup)
        self.assertIn("sha256sum --check --status", setup)
        self.assertIn("localization_checkpoint.pth.part", setup)
        self.assertIn("mv --", setup)
        self.assertIn("trap", setup)

    def test_training_uses_default_localization_checkpoint_when_not_provided(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("default_localisation_checkpoint", train)
        self.assertIn("cmd_args.get('localisation') or default_localisation_checkpoint", train)

    def test_inference_uses_default_localization_checkpoint_when_not_provided(self):
        infer = (MODEL_DIR / "infer.py").read_text(encoding="utf-8")
        self.assertIn("default_localisation_checkpoint", infer)
        self.assertIn("cmd_args.get(\"localisation\") or default_localisation_checkpoint", infer)

    def test_localization_option_is_a_checkpoint_file_and_optional(self):
        schema = __import__("json").loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
        option = schema["properties"]["localisation"]
        self.assertEqual(option["format"], "file:localization_checkpoint.pth")
        self.assertNotIn("localisation", schema.get("required", []))

    def test_training_logs_nested_diagnostics_and_limits_sample_count(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("training_artifact_path", train)
        self.assertIn("visualization_samples", train)
        self.assertIn("visualized >=", train)

    def test_training_visualizes_train_and_validation_loaders_separately(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("self.validation_dataset_it", train)
        self.assertIn("validation_kwargs['split'] = 'test'", train)
        self.assertIn("self.visualize(self.train_dataset_it, epoch, 'training')", train)
        self.assertIn("self.visualize(self.validation_dataset_it, epoch, 'validation')", train)

    def test_checkpoints_are_stored_under_checkpoint_subfolder(self):
        train = (MODEL_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn('"checkpoints", "checkpoint.pth"', train)
        self.assertIn("artifacts/checkpoints/checkpoint.pth", train)


if __name__ == "__main__":
    unittest.main()
