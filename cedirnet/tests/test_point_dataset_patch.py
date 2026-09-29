"""Exercise the dataset implementation installed by the plugin patch."""

import ast
import pathlib
import tempfile
import unittest

import numpy as np


MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]


def patched_dataset_source():
    patch = (MODEL_DIR / "0002-Add-GenericDataset.patch").read_text()
    hunk = patch.split("@@ -0,0 +1,", 1)[1].split("\n", 1)[1]
    hunk = hunk.split("diff --git", 1)[0]
    return "\n".join(line[1:] for line in hunk.splitlines() if line.startswith("+")) + "\n"


class PointDatasetPatchTest(unittest.TestCase):
    def test_point_and_vector_annotations_use_correct_centers_and_direction(self):
        tree = ast.parse(patched_dataset_source())
        dataset = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GenericDataset")
        getitem = next(node for node in dataset.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
        loop = next(node for node in ast.walk(getitem) if isinstance(node, ast.For) and
                    isinstance(node.target, ast.Name) and node.target.id == "point")
        code = compile(ast.Module(body=[loop], type_ignores=[]), "GenericDataset.py", "exec")

        class Tensor:
            def __init__(self, value):
                self.value = value
            def __setitem__(self, key, value):
                self.value[key] = value

        def run(points, scale=1.0):
            namespace = dict(annot=points, np=np, M=2, instance_counter=1,
                             centers=[], orientation=Tensor(np.zeros((1, 32, 32))),
                             label=Tensor(np.zeros((32, 32))),
                             instances=Tensor(np.zeros((32, 32))),
                             self=type("Dataset", (), {"resize_factor": scale})())
            exec(code, namespace)
            return namespace

        point = run([[8, 9]])
        np.testing.assert_array_equal(point["centers"], [[8, 9]])
        self.assertEqual(point["orientation"].value[0, 9, 8], 0)
        self.assertEqual(point["instances"].value[9, 8], 1)
        scaled = run([[8, 9]], 2.0)
        np.testing.assert_array_equal(scaled["centers"], [[16, 18]])
        vector = run([[8, 9, 8, 13]])
        self.assertAlmostEqual(vector["orientation"].value[0, 9, 8], np.pi / 2)
        with self.assertRaisesRegex(ValueError, "Expected"):
            run([[8, 9, 10]])

    def test_training_explicitly_disables_orientation(self):
        source = (MODEL_DIR / "train.py").read_text()
        self.assertIn("if cmd_args.get('orientation', False):", source)
        self.assertIn("enable_3dof=False", source)


if __name__ == "__main__":
    unittest.main()
