import importlib.util
import json
import pathlib
import tempfile
import unittest

from PIL import Image


MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        raise AssertionError(f"missing CeDiRNet-STEM module: {filename}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load CeDiRNet-STEM module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnnotationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotations = load_module("cedirnet_stem_annotations", "annotations.py")

    def test_direct_point_radius_annotation(self):
        point = self.annotations.parse_point_radius([12, 20, 7])
        self.assertEqual(point, (12.0, 20.0, 7.0))

    def test_vector_annotation_uses_distance_as_radius(self):
        point = self.annotations.parse_point_radius([12, 20, 15, 24])
        self.assertEqual(point[:2], (12.0, 20.0))
        self.assertAlmostEqual(point[2], 5.0)

    def test_non_positive_radius_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "radius must be positive"):
            self.annotations.parse_point_radius([12, 20, 0])

    def test_out_of_bounds_center_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside image bounds"):
            self.annotations.build_targets(10, 8, [[10, 4, 2]])

    def test_targets_contain_center_instance_and_radius(self):
        targets = self.annotations.build_targets(20, 16, [[8, 6, 4]], support_radius=2)
        self.assertEqual(targets["centers"], [(8.0, 6.0)])
        self.assertEqual(targets["instance"].shape, (16, 20))
        self.assertEqual(targets["shape_coef"].shape, (1, 16, 20))
        self.assertEqual(int(targets["instance"][6, 8]), 1)
        self.assertAlmostEqual(float(targets["shape_coef"][0, 6, 8]), 4.0)
        self.assertEqual(int(targets["label"][6, 8]), 1)

    def test_stem_channels_are_bf_haadf_and_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bf_path = root / "particle_BF.png"
            haadf_path = root / "particle_HAADF.png"
            Image.new("L", (3, 2), color=17).save(bf_path)
            Image.new("L", (3, 2), color=91).save(haadf_path)

            image = self.annotations.load_stem_image(bf_path)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((1, 1)), (17, 91, 0))

    def test_missing_haadf_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bf_path = pathlib.Path(directory) / "particle_BF.png"
            Image.new("L", (3, 2), color=17).save(bf_path)
            with self.assertRaisesRegex(FileNotFoundError, "HAADF"):
                self.annotations.load_stem_image(bf_path)

    def test_haadf_primary_still_composes_bf_then_haadf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bf_path = root / "particle_BF.png"
            haadf_path = root / "particle_HAADF.png"
            Image.new("L", (3, 2), color=17).save(bf_path)
            Image.new("L", (3, 2), color=91).save(haadf_path)

            image = self.annotations.load_stem_image(haadf_path)
            self.assertEqual(image.getpixel((1, 1)), (17, 91, 0))


class InferenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_module("cedirnet_stem_results", "results.py")

    def test_resize_prediction_returns_original_pixel_radius(self):
        center, radius = self.results.restore_prediction(
            center=(50, 25),
            radius=10,
            network_size=(100, 50),
            original_size=(200, 100),
        )
        self.assertEqual(center, (0.5, 0.5))
        self.assertEqual(radius, 20.0)

    def test_anisotropic_resize_uses_inverse_training_radius_scale(self):
        center, radius = self.results.restore_prediction(
            center=(50, 50),
            radius=7.5,
            network_size=(100, 100),
            original_size=(200, 100),
        )
        self.assertEqual(center, (0.5, 0.5))
        self.assertEqual(radius, 10.0)

    def test_label_studio_result_is_center_to_radius_vector(self):
        result = self.results.label_studio_vector_result(
            center=(0.25, 0.5),
            radius=10,
            score=0.8,
            original_size=(200, 100),
            from_name="labels",
            to_name="image",
            label="Particle",
            result_id="particle-1",
        )
        self.assertEqual(result["type"], "labels")
        vertices = result["value"]["vertices"]
        self.assertEqual(vertices[0]["x"], 25.0)
        self.assertEqual(vertices[0]["y"], 50.0)
        self.assertEqual(vertices[1]["x"], 30.0)
        self.assertEqual(vertices[1]["y"], 50.0)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_label_studio_radius_handle_stays_inside_right_edge(self):
        result = self.results.label_studio_vector_result(
            center=(0.95, 0.5),
            radius=10,
            score=0.8,
            original_size=(100, 100),
            from_name="labels",
            to_name="image",
            label="Particle",
            result_id="particle-2",
        )
        vertices = result["value"]["vertices"]
        self.assertEqual(vertices[0]["x"], 95.0)
        self.assertEqual(vertices[1]["x"], 85.0)


class CheckpointCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoints = load_module("cedirnet_stem_checkpoint", "checkpoint.py")

    def test_only_matching_checkpoint_tensors_are_loaded(self):
        class Tensor:
            def __init__(self, shape):
                self.shape = shape

        class Model:
            def __init__(self):
                self.loaded = None

            def state_dict(self):
                return {"matching": Tensor((2, 3)), "different": Tensor((4,))}

            def load_state_dict(self, state, strict):
                self.loaded = (state, strict)
                return [], []

        model = Model()
        skipped = self.checkpoints.load_compatible_model_state(
            model,
            {
                "model_state_dict": {
                    "matching": Tensor((2, 3)),
                    "different": Tensor((5,)),
                    "upstream_extra_head": Tensor((1,)),
                }
            },
        )
        self.assertEqual(set(model.loaded[0]), {"matching"})
        self.assertFalse(model.loaded[1])
        self.assertEqual(skipped, {"different", "upstream_extra_head"})

    def test_checkpoint_without_model_weights_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_state_dict"):
            self.checkpoints.load_compatible_model_state(object(), {})


class PreparedModelFilesTest(unittest.TestCase):
    def test_model_schema_documents_point_radius_manifest(self):
        model_json = MODEL_DIR / "model.json"
        self.assertTrue(model_json.exists(), "missing model.json")
        schema = json.loads(model_json.read_text())
        description = schema["properties"]["manifest"]["description"]
        self.assertIn("[x, y, radius]", description)
        self.assertIn("[x, y, radius_x, radius_y]", description)
        self.assertIn("BF", description)
        self.assertIn("HAADF", description)

    def test_label_config_uses_vector_radius_handle(self):
        config = (MODEL_DIR / "config.yml").read_text()
        self.assertIn('<Vector name="radius"', config)
        self.assertIn("center", config.lower())
        self.assertIn("radius", config.lower())

    def test_setup_clones_stem_repository(self):
        setup = (MODEL_DIR / "setup.sh").read_text()
        self.assertIn("vicoslab/CeDiRNet-STEM", setup)
        self.assertIn("GenericPointRadiusDataset.py", setup)
        self.assertNotIn("vicoslab/toolbox.git", setup)

    def test_training_config_is_radius_only(self):
        config = (MODEL_DIR / "base_config.py").read_text()
        self.assertIn("NUM_VECTOR_FIELDS = 4", config)
        self.assertIn('"name": "generic_point_radius"', config)
        self.assertIn('"shape_type": "circle"', config)

    def test_training_entrypoint_uses_manifest(self):
        train = (MODEL_DIR / "train.py").read_text()
        self.assertIn('cmd_args["manifest"]', train)
        self.assertIn("GenericPointRadiusDataset", (MODEL_DIR / "setup.sh").read_text())
        self.assertIn("mlflow", train)

    def test_inference_returns_radius_and_label_studio_vectors(self):
        infer = (MODEL_DIR / "infer.py").read_text()
        self.assertIn('"radii"', infer)
        self.assertIn("label_studio_vector_result", infer)
        self.assertIn("pred_attributes", infer)
        self.assertIn("load_stem_image", infer)
        self.assertIn('ARGS["model"]["kwargs"]["pretrained"] = False', infer)

    def test_browser_ui_draws_radius_circles(self):
        ui = (MODEL_DIR / "ui.html").read_text()
        self.assertIn("radii", ui)
        self.assertIn("borderRadius", ui)


if __name__ == "__main__":
    unittest.main()
