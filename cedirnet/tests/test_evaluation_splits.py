"""Exercise actual Trainer methods with CPU tensors and a local MLflow store.

The heavyweight upstream imports are bypassed using AST extraction; dataset
routing, evaluation, scheduling and metric logging execute the production code.
"""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mlflow
import numpy as np
import scipy.optimize
import scipy.spatial
import torch
from tqdm import tqdm

MODEL_DIR = Path(__file__).resolve().parents[1]


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def match_centers(predictions, targets):
    distances = scipy.spatial.distance_matrix(predictions, targets)
    costs = distances.copy()
    costs[costs > 20] = np.finfo(float).max
    rows, cols = scipy.optimize.linear_sum_assignment(1 / (costs + 1e-10), maximize=True)
    valid = distances[rows, cols] < 20
    return rows[valid], cols[valid]


class ManifestDataset(torch.utils.data.Dataset):
    """Small labeled fixtures, including negatives, using GenericDataset routing."""
    def __init__(self, manifest, split, **kwargs):
        data = json.loads(Path(manifest).read_text())
        items = data.get(split, data.get('data', []) if split == 'train' else [])
        self.items = [item for item in items if 'points' in item]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {'name': item['image_path'], 'image': torch.zeros(3, 8, 8),
                'center': torch.tensor(item['points'] or [[0, 0]], dtype=torch.float32),
                'orientation': torch.zeros(1, 8, 8)}


def collate(samples):
    # The test fixtures have one center/sentinel per image.
    return {key: [s[key] for s in samples] if key == 'name'
            else torch.stack([s[key] for s in samples]) for key in samples[0]}


class CenterModel(torch.nn.Module):
    def __init__(self, columns=6):
        super().__init__()
        self.instance_center_estimator = torch.nn.Identity()
        self.instance_center_estimator.local_max_thr = .1
        self.columns = columns
        self.batch_sizes = []

    def forward(self, output, **sample):
        image = output
        assert not self.training and not torch.is_grad_enabled()
        self.batch_sizes.append(len(image))
        rows = torch.zeros(len(image), 2, self.columns)
        # True positive score .9, false positive .2: eval's appended constant
        # must not be mistaken for localization confidence (column 4).
        rows[:, :, 0] = 1
        rows[:, 0, 1:3] = torch.tensor([2, 3])
        rows[:, 1, 1:3] = torch.tensor([6, 6])
        rows[:, :, 4] = torch.tensor([.9, .2])
        if self.columns > 5:
            rows[:, :, 5:] = 1
        return {'output': image[:, :2], 'center_pred': rows,
                'center_heatmap': torch.zeros(len(image), 1, 8, 8),
                'pred_angle': torch.zeros(len(image), 2)}


class EvaluationSplitsTest(unittest.TestCase):
    def setUp(self):
        metrics = load_helper('validation_metrics')
        self.namespace = {'torch': torch, 'np': np, 'tqdm': tqdm, 'mlflow': mlflow,
                          'POINT_MATCH_DISTANCE_PX': metrics.POINT_MATCH_DISTANCE_PX,
                          'ValidationMetrics': metrics.ValidationMetrics,
                          'extract_ground_truth': metrics.extract_ground_truth,
                          'center_detection_threshold': load_helper('detection_threshold').center_detection_threshold,
                          'CenterGlobalMinimizationEval': lambda **kw: SimpleNamespace(
                              _assign_detections_to_groundtruth=match_centers),
                          'should_validate': load_helper('scheduling').should_validate,
                          'get_centerdir_dataset': lambda name, kwargs, opts, **extra: (
                              ManifestDataset(**kwargs), None),
                          'variable_len_collate': collate}
        mode = next(n for n in ast.parse((MODEL_DIR / 'extras.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'set_center_model_mode')
        exec(compile(ast.Module(body=[mode], type_ignores=[]), 'extras.py', 'exec'), self.namespace)
        tree = ast.parse((MODEL_DIR / 'train.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(MODEL_DIR / 'train.py'), 'exec'), self.namespace)
        self.trainer = self.namespace['Trainer']({'cuda': False, 'visualization_samples': 1,
                                                 'validation_score_threshold': .5,
                                                 'train_dataset': {}})
        self.trainer.dataset_batch = 2
        self.trainer.centerdir_groundtruth_op = None
        self.trainer.model = torch.nn.Identity()
        self.trainer.center_model = CenterModel()
        self.trainer.visualize_sample = Mock()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest = Path(self.tmp.name) / 'manifest.json'

    def initialize(self, data):
        self.manifest.write_text(json.dumps(data))
        self.trainer.initialize_evaluation_loaders({'manifest': str(self.manifest)})

    def test_data_only_manifest_still_evaluates_entire_training_set(self):
        entries = [{'image_path': f'{i}.png', 'points': [[2, 3]]} for i in range(3)]
        entries += [{'image_path': 'negative.png', 'points': []}, {'image_path': 'unlabeled.png'}]
        self.initialize({'data': entries, 'version': 4})
        with patch.object(mlflow, 'log_metrics') as log, contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.trainer.evaluate_splits(9)
        self.assertEqual(list(result), ['training'])
        self.assertEqual(result['training']['training/images'], 4)
        self.assertEqual(result['training']['training/point_tp'], 3)
        self.assertEqual(result['training']['training/point_fp'], 1)
        self.assertEqual(self.trainer.center_model.batch_sizes, [2, 2])
        self.assertEqual(log.call_count, 1)
        self.assertEqual(log.call_args.kwargs['step'], 10)
        self.assertEqual(self.trainer.visualize_sample.call_count, 1)
        self.assertIn('Validation: skipped', output.getvalue())
        self.assertIn('Testing: skipped', output.getvalue())
        self.assertIn('Training: ', output.getvalue())
        self.assertEqual(self.trainer.center_model.instance_center_estimator.local_max_thr, .1)

    def test_train_val_test_metrics_are_distinct_in_real_mlflow(self):
        self.initialize({key: [{'image_path': f'{key}-{i}.png', 'points': [[2, 3]]}
                               for i in range(size)]
                         for key, size in [('train', 3), ('val', 2), ('test', 1)]})
        old_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, old_uri)
        mlflow.set_tracking_uri(f'sqlite:///{self.tmp.name}/mlflow.db')
        experiment = mlflow.create_experiment('split-regression', artifact_location=Path(self.tmp.name).as_uri())
        with mlflow.start_run(experiment_id=experiment) as run, contextlib.redirect_stdout(io.StringIO()):
            result = self.trainer.evaluate_splits(6)
        client = mlflow.MlflowClient()
        actual = client.get_run(run.info.run_id).data.metrics
        self.assertFalse(any('/orientation_' in key for key in actual))
        self.assertEqual(set(result), {'training', 'validation', 'testing'})
        for subset, count in [('training', 3), ('validation', 2), ('testing', 1)]:
            self.assertEqual(actual[f'{subset}/images'], count)
            self.assertEqual(actual[f'{subset}/point_f1_at_20px'], 1)
            history = client.get_metric_history(run.info.run_id, f'{subset}/images')
            self.assertEqual([(m.step, m.value) for m in history], [(7, count)])
        self.assertEqual(self.trainer.center_model.batch_sizes, [2, 1, 2, 1])
        self.assertEqual([c.kwargs['subset'] for c in self.trainer.visualize_sample.call_args_list],
                         ['training', 'validation', 'validation', 'testing'])

    def test_enabled_training_metrics_cover_all_samples_beyond_figure_limit(self):
        self.trainer.args.update(evaluate_training=True, visualization_samples=1)
        self.initialize({'data': [{'image_path': f'{i}.png', 'points': [[2, 3]]}
                                  for i in range(7)]})
        with patch.object(mlflow, 'log_metrics'), contextlib.redirect_stdout(io.StringIO()):
            values = self.trainer.evaluate_splits(0)['training']
        self.assertEqual(values['training/images'], 7)
        self.assertEqual(values['training/ground_truth_points'], 7)
        self.assertEqual(values['training/point_tp'], 7)
        self.assertEqual(self.trainer.center_model.batch_sizes, [2, 2, 2, 1])
        self.assertEqual(self.trainer.visualize_sample.call_count, 1)

    def test_empty_explicit_train_does_not_fall_back_to_unassigned_data(self):
        self.initialize({'train': [], 'data': [{'image_path': 'x.png', 'points': [[2, 3]]}],
                         'val': [], 'test': [{'image_path': 'unlabeled.png'}]})
        with patch.object(mlflow, 'log_metrics') as log, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.trainer.evaluate_splits(0), {})
        log.assert_not_called()
        self.assertIn('Training: skipped', output.getvalue())

    def test_score_column_is_localization_confidence_for_all_row_schemas(self):
        self.initialize({'data': [{'image_path': 'positive.png', 'points': [[2, 3]]}]})
        for columns in [5, 6, 7]:
            with self.subTest(columns=columns):
                self.trainer.center_model = CenterModel(columns)
                with patch.object(mlflow, 'log_metrics'), contextlib.redirect_stdout(io.StringIO()):
                    result = self.trainer.evaluate_splits(0)['training']
                self.assertEqual(result['training/point_fp'], 0)
                self.assertEqual(result['training/point_f1_at_20px'], 1)

    def test_nondefault_interval_and_final_epoch_evaluate_all_splits(self):
        self.initialize({'data': [{'image_path': 'positive.png', 'points': [[2, 3]]}]})
        self.trainer.args.update(n_epochs=10, display=True, display_it=3, save=False)
        self.trainer.train = Mock(return_value=0)
        self.trainer.scheduler = self.trainer.center_scheduler = None
        with patch.object(self.trainer, 'evaluate_splits', wraps=self.trainer.evaluate_splits) as evaluate:
            with patch.object(mlflow, 'log_metrics') as log, contextlib.redirect_stdout(io.StringIO()):
                self.trainer.run()
        self.assertEqual([c.args[0] for c in evaluate.call_args_list], [2, 5, 8, 9])
        self.assertEqual([c.kwargs['step'] for c in log.call_args_list], [3, 6, 9, 10])

    def test_visualizations_use_localization_score_and_xy_only(self):
        for columns in [5, 6, 7]:
            with self.subTest(columns=columns):
                centers = np.zeros((2, columns), dtype=np.float32)
                centers[:, 0] = 1
                centers[:, 1:3] = [[2, 3], [6, 6]]
                centers[:, 4] = [.9, .2]
                if columns > 5:
                    centers[:, 5:] = 1
                plot = Mock(return_value=object())
                self.namespace.update(plot_training_diagnostics=lambda **kw: (plot(**kw), None),
                                      log_figure_artifact=Mock(),
                                      training_artifact_path=Mock(return_value='probe.png'),
                                      plt=SimpleNamespace(close=Mock()))
                self.namespace['Trainer'].visualize_sample(
                    self.trainer, 0, 'training', 'probe.png', np.zeros((3, 8, 8)),
                    centers, np.zeros(2), np.zeros((2, 8, 8)),
                    np.zeros((8, 8)), np.array([[2, 3]]), detection_score_threshold=.5)
                np.testing.assert_array_equal(plot.call_args.kwargs['centers'], [[2, 3]])
                np.testing.assert_allclose(plot.call_args.kwargs['scores'], [.9])

    def test_point_only_evaluation_omits_unsupervised_orientation_metrics(self):
        self.initialize({key: [{'image_path': f'{key}.png', 'points': [[2, 3]]}]
                         for key in ['train', 'val', 'test']})
        with patch.object(mlflow, 'log_metrics') as log, contextlib.redirect_stdout(io.StringIO()) as output:
            results = self.trainer.evaluate_splits(0)
        self.assertEqual(log.call_count, 3)
        for values in results.values():
            self.assertFalse(any('/orientation_' in key for key in values))
            self.assertTrue(any('/localization_mae_px' in key for key in values))
        self.assertNotIn('orientation MAE', output.getvalue())

    def test_disabled_training_evaluation_keeps_held_out_evaluation(self):
        self.trainer.args['evaluate_training'] = False
        factory = self.namespace['get_centerdir_dataset']
        with patch.dict(self.namespace, get_centerdir_dataset=Mock(wraps=factory)):
            self.initialize({key: [{'image_path': f'{key}.png', 'points': [[2, 3]]}]
                             for key in ['train', 'val', 'test']})
            requested = [c.args[1]['split'] for c in self.namespace['get_centerdir_dataset'].call_args_list]
        self.assertEqual(requested, ['train', 'val', 'test'])
        with patch.object(mlflow, 'log_metrics') as log, contextlib.redirect_stdout(io.StringIO()) as output:
            results = self.trainer.evaluate_splits(0)
        self.assertEqual(set(results), {'validation', 'testing'})
        self.assertEqual(log.call_count, 2)
        self.assertEqual([c.kwargs['subset'] for c in self.trainer.visualize_sample.call_args_list],
                         ['training', 'validation', 'testing'])
        self.assertIn('Training evaluation: skipped (disabled by user)', output.getvalue())

    def test_disabled_training_evaluation_still_visualizes_limited_data_only_samples(self):
        self.trainer.args.update(evaluate_training=False, visualization_samples=3)
        self.initialize({'data': [{'image_path': f'{i}.png', 'points': [[2, 3]]}
                                  for i in range(7)]})
        with patch.object(mlflow, 'log_metrics') as log, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.trainer.evaluate_splits(0), {})
        log.assert_not_called()
        self.assertEqual(self.trainer.visualize_sample.call_count, 3)
        self.assertEqual(self.trainer.center_model.batch_sizes, [2, 2])
        self.assertTrue(all(c.kwargs['subset'] == 'training'
                            for c in self.trainer.visualize_sample.call_args_list))
        self.assertEqual(self.trainer.center_model.instance_center_estimator.local_max_thr, .1)

    def test_training_evaluation_option_defaults_true_and_cli_false_is_propagated(self):
        schema = json.loads((MODEL_DIR / 'model.json').read_text())
        option = schema['properties']['evaluate_training']
        self.assertEqual(option['type'], 'boolean')
        self.assertIs(option['default'], True)
        self.assertEqual(option['stage'], 'train')
        toolbox = MODEL_DIR.parents[1] / 'toolbox/apps/modelargs/modelargs/__init__.py'
        spec = importlib.util.spec_from_file_location('evaluation_modelargs', toolbox)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import sys
        tree = ast.parse((MODEL_DIR / 'train.py').read_text())
        main = tree.body[-1]
        assignments = [node for node in main.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                               and t.value.id == 'args' and isinstance(t.slice, ast.Constant)
                               and t.slice.value == 'evaluate_training' for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        # Isolate this option from the existing parser's unrelated required-field
        # bug; retain the exact production option schema and real boolean parser.
        parser_schema = Path(self.tmp.name) / 'option-schema.json'
        parser_schema.write_text(json.dumps(dict(schema, properties={'evaluate_training': option})))
        for flags, expected in [([], True), (['--evaluate_training', 'false'], False)]:
            with patch.object(sys, 'argv', ['train.py', *flags]):
                parsed = module.parse(str(parser_schema))
            self.assertIs(parsed['evaluate_training'], expected)
            namespace = {'args': {}, 'cmd_args': parsed}
            exec(compile(ast.Module(body=assignments, type_ignores=[]), 'train.py', 'exec'), namespace)
            self.assertIs(namespace['args']['evaluate_training'], expected)

    def test_training_losses_are_namespaced_without_changing_values(self):
        import pandas as pd
        tree = ast.parse((MODEL_DIR / 'train.py').read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        train = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        statements = [n for n in train.body if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Attribute)
                      and n.value.func.attr == 'log_metrics']
        self.assertEqual(len(statements), 1)
        with patch.object(mlflow, 'log_metrics') as log:
            exec(compile(ast.Module(body=statements, type_ignores=[]), 'train.py', 'exec'),
                 {'mlflow': mlflow, 'pd': pd, 'all_metrics': [{'loss': 2., 'centerdir_total': 1.},
                                                           {'loss': 4., 'centerdir_total': 3.}], 'epoch': 8})
        self.assertEqual(log.call_args.args[0], {'training/loss': 3., 'training/centerdir_total': 2.})
        self.assertEqual(log.call_args.kwargs['step'], 9)

    def test_testing_artifact_path_is_supported(self):
        diagnostics = load_helper('diagnostics')
        self.assertEqual(diagnostics.training_artifact_path(0, '../a.png', 'testing'),
                         'visualizations/epoch_0001/testing/a-diagnostics.png')


if __name__ == '__main__':
    unittest.main()
