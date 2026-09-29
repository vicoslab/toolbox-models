"""Execute training lifecycle code without loading CUDA/model dependencies.

Set TOOLBOX_SOURCE to a Toolbox checkout to test its actual modelargs emitter
and Nexus log parser (TOOLBOX_REV defaults to HEAD).
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch


TRAIN = Path(__file__).resolve().parents[1] / "train.py"


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), "exec"), namespace)


class TrainingActionsTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(TRAIN.read_text())
        self.info = SimpleNamespace(experiment_id="17", run_id="test-run")
        self.active_run = SimpleNamespace(info=self.info)
        self.actions = []
        self.namespace: dict[str, Any] = {
            "modelargs": SimpleNamespace(emit_action=lambda *args: self.actions.append(args)),
            "run": self.active_run,
        }
        self.host_parser = None
        if source := os.environ.get("TOOLBOX_SOURCE"):
            def read(path):
                return subprocess.check_output(
                    ["git", "-C", source, "show", f"{os.getenv('TOOLBOX_REV', 'HEAD')}:{path}"],
                    text=True,
                )
            modelargs = {}
            exec(compile(read("apps/modelargs/modelargs/__init__.py"), "modelargs", "exec"), modelargs)
            host = ast.parse(read("apps/nexus/nexus.py"))
            nodes = [node for node in host.body if (
                isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in {"tqdm_header", "quick_action"}
                    for t in node.targets
                )
            ) or (isinstance(node, ast.FunctionDef) and node.name == "refresh_logs")]
            parser: dict[str, Any] = {"re": re}
            execute(nodes, parser)
            self.host_parser = parser["refresh_logs"]

            def emit(kind, value):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    modelargs["emit_action"](kind, value)
                proc = SimpleNamespace(stdout=io.StringIO(stream.getvalue()), poll=lambda: None)
                task = {"process": proc, "output": []}
                self.host_parser(task)
                self.actions.extend(task["output"])
            self.namespace["modelargs"].emit_action = emit

    def test_run_announced_before_model_initialization(self):
        main = self.tree.body[-1]
        assert isinstance(main, ast.If)
        run_context = next(n for n in main.body if isinstance(n, ast.With))
        # Execute startup through Trainer construction. Signal/MLflow operations
        # are mocked; the actual notification statements are not replaced.
        self.namespace.update(
            signal=SimpleNamespace(signal=Mock(), SIGINT=2, SIGTERM=15),
            mlflow=SimpleNamespace(log_params=Mock(), log_param=Mock()),
            json=__import__("json"), args={}, cmd_args={},
            POINT_MATCH_DISTANCE_PX=20,
        )
        def trainer(args):
            self.assertEqual(self.actions, [("Experiment", "17"), ("Run", "test-run")])
            return SimpleNamespace(initialize=Mock(), run=Mock())
        self.namespace["Trainer"] = trainer
        execute(run_context.body, self.namespace)

    def checkpoint(self, local, fail=False):
        cls = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        execute([method], self.namespace)
        trainer = SimpleNamespace(
            args={"n_epochs": 2, "display": False, "save": True, "save_interval": 2},
            train=Mock(return_value=0), scheduler=None, center_scheduler=None,
            model=SimpleNamespace(state_dict=lambda: {}), center_model=None,
        )
        saved = []
        uploaded = []
        def save(state, filename):
            self.assertEqual(self.actions, [])  # no premature inference link
            if fail:
                raise OSError("checkpoint write failed")
            Path(filename).write_bytes(b"test checkpoint")
            saved.append(filename)
        def upload(filename, artifact_path):
            self.assertEqual(self.actions, [])  # upload must finish before announcement
            self.assertTrue(Path(filename).is_file())
            uploaded.append(artifact_path)
        self.namespace.update(
            os=os, tempfile=tempfile, torch=SimpleNamespace(save=save),
            mlflow=SimpleNamespace(active_run=lambda: self.active_run, log_artifact=upload),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MLFLOW_ARTIFACTS_DESTINATION": directory if local else ""}):
                if fail:
                    with self.assertRaises(OSError):
                        self.namespace["run"](trainer)
                    self.assertEqual(self.actions, [])
                    return
                self.namespace["run"](trainer)
            self.assertEqual(len(saved), 1)
            self.assertEqual(self.actions, [("Weights", "mlflow-artifacts:/17/test-run/artifacts/checkpoints/checkpoint.pth")])
            if local:
                self.assertTrue(Path(directory, "17/test-run/artifacts/checkpoints/checkpoint.pth").is_file())
                self.assertEqual(uploaded, [])
            else:
                self.assertEqual(uploaded, ["checkpoints"])

    def test_local_checkpoint_announces_inference_after_save(self):
        self.checkpoint(local=True)

    def test_remote_checkpoint_announces_inference_after_upload(self):
        self.checkpoint(local=False)

    def test_failed_checkpoint_does_not_announce_inference(self):
        self.checkpoint(local=True, fail=True)


if __name__ == "__main__":
    unittest.main()
