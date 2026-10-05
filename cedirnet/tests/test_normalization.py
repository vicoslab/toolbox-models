"""Real localizer regressions. Set CEDIRNET_SOURCE and CEDIRNET_TEST_LOCALIZATION.

Run in a separate process from STEM tests (both upstreams import `models`).
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch

PLUGIN = Path(__file__).resolve().parents[1]
SOURCE = os.getenv('CEDIRNET_SOURCE')
CHECKPOINT = os.getenv('CEDIRNET_TEST_LOCALIZATION')


class InferenceBoundaryTest(unittest.TestCase):
    def test_inference_disables_autograd(self):
        import numpy as np
        enabled = []
        def model(tensor):
            enabled.append(torch.is_grad_enabled())
            return tensor
        def center(tensor, **kwargs):
            return {'center_pred': torch.zeros(1, 1, 5), 'pred_angle': torch.zeros(1, 1, 1)}
        tree = ast.parse((PLUGIN / 'infer.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
        namespace = {'torch': torch, 'np': np, 'model': model, 'center_model': center, 'width': 128, 'height': 128}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'infer.py', 'exec'), namespace)
        with torch.enable_grad():
            namespace['predict'](torch.zeros(1, 3, 128, 128))
        self.assertEqual(enabled, [False])


@unittest.skipUnless(SOURCE and CHECKPOINT, 'set source and official localization checkpoint')
class NormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SOURCE is not None
        sys.path[:0] = [str(PLUGIN), str(Path(SOURCE) / 'src')]
        from base_config import get_args
        from models import get_center_model
        spec = importlib.util.spec_from_file_location('normalization_extras', PLUGIN / 'extras.py')
        cls.extras = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.extras)
        cls.get_args = staticmethod(get_args)
        cls.get_center_model = staticmethod(get_center_model)
        torch.set_num_threads(2)

    def setUp(self):
        self.args = self.get_args(128, 128, enable_3dof=False)
        self.state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        size = 128
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        def field(dx, dy, scale):
            radius = (dx.square() + dy.square()).sqrt().clamp_min(1)
            zero = torch.zeros_like(radius)
            return torch.stack([dx/radius*scale, dy/radius*scale, zero, zero, zero])[None]
        self.negative = field((16-x % 32).float(), (16-y % 32).float(), .03)
        self.positive = field((size//2-x).float(), (size//2-y).float(), 1)

    def loaded(self):
        return self.extras.load_center_model(self.args, self.state, 'cpu')

    def batchnorm(self, center):
        return [m for m in center.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    def test_loader_retains_checkpoint_statistics_without_editing_arguments_or_state(self):
        kwargs = copy.deepcopy(self.args['center_model']['kwargs'])
        first = self.state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'].clone()
        center = self.loaded()
        self.assertEqual(self.args['center_model']['kwargs'], kwargs)
        torch.testing.assert_close(self.state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'], first)
        for name, module in center.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertTrue(module.track_running_stats)
                torch.testing.assert_close(module.running_mean, self.state['center_model_state_dict'][name+'.running_mean'])
                torch.testing.assert_close(module.running_var, self.state['center_model_state_dict'][name+'.running_var'])

    def test_eval_heatmaps_are_independent_of_companion_image(self):
        center = self.loaded().eval()
        with torch.no_grad():
            alone = center(self.negative.clone())
            positive_alone = center(self.positive.clone())
            mixed = center(torch.cat([self.negative, self.positive]))
            reversed_batch = center(torch.cat([self.positive, self.negative]))
        for out, negative_index, positive_index in ((mixed, 0, 1), (reversed_batch, 1, 0)):
            torch.testing.assert_close(alone['center_heatmap'][0], out['center_heatmap'][negative_index], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(positive_alone['center_heatmap'][0], out['center_heatmap'][positive_index], rtol=1e-5, atol=1e-6)
            self.assertEqual(int((out['center_pred'][negative_index, :, 4] >= .5).sum()), 0)
            expected = positive_alone['center_pred'][0]
            expected = expected[expected[:, 4] >= .5]
            actual = out['center_pred'][positive_index]
            actual = actual[actual[:, 4] >= .5]
            self.assertGreater(len(expected), 0)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(int((alone['center_pred'][0, :, 4] >= .5).sum()), 0)

    def test_training_preserves_original_batch_statistics_without_updating_eval_buffers(self):
        center = self.loaded()
        baseline_kwargs = copy.deepcopy(self.args['center_model']['kwargs'])
        baseline = self.get_center_model(self.args['center_model']['name'], baseline_kwargs, is_learnable=True)
        baseline.init_output(self.args['num_vector_fields'])
        baseline = torch.nn.DataParallel(baseline)
        values = dict(self.state['center_model_state_dict'])
        values['module.instance_center_estimator.conv_start.0.weight'] = values['module.instance_center_estimator.conv_start.0.weight'][:, :2]
        baseline.load_state_dict(values, strict=False)
        baseline.train()
        before = {k:v.clone() for k,v in center.named_buffers()}
        # Public helper must leave the wrapper in train mode (raw regression).
        self.extras.set_center_model_mode(center, training=True, freeze_learning=True)
        with torch.no_grad():
            actual = center(torch.cat([self.negative, self.positive]))
            expected = baseline(torch.cat([self.negative, self.positive]))
        self.assertTrue(center.module.training)
        torch.testing.assert_close(actual['output'], torch.cat([self.negative, self.positive]), rtol=0, atol=0)
        torch.testing.assert_close(actual['center_heatmap'], expected['center_heatmap'], rtol=0, atol=0)
        for name, value in center.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.extras.set_center_model_mode(center, training=False)
        with torch.no_grad():
            alone = center(self.negative.clone())
            mixed = center(torch.cat([self.negative, self.positive]))
        torch.testing.assert_close(alone['center_heatmap'][0], mixed['center_heatmap'][0], rtol=1e-5, atol=1e-6)

    def test_validation_reenables_fixed_statistics_after_training(self):
        import importlib
        from unittest.mock import patch
        from types import SimpleNamespace
        train = importlib.import_module('train')
        center = self.loaded()
        self.extras.set_center_model_mode(center, training=True, freeze_learning=True)
        trainer = train.Trainer({'cuda': False, 'validation_score_threshold': .5})
        trainer.model = torch.nn.Identity()
        trainer.center_model = center
        sample = {
            'image': torch.cat([self.negative, self.positive]),
            'orientation': torch.zeros(2, 1, 128, 128),
            'center': torch.zeros(2, 4, 2),
            'name': ['negative', 'positive'],
        }
        class Loader:
            dataset = [0, 1]
            def __iter__(self):
                yield sample
        trainer.evaluation_loaders = {'validation': Loader()}
        rendered = []
        trainer.visualize_sample = lambda **kw: rendered.append(kw['localization_response'])
        with patch.object(train.mlflow, 'log_metrics'):
            trainer.evaluate(0, 'validation')
        self.assertTrue(all(m.track_running_stats for m in self.batchnorm(center)))
        with torch.no_grad():
            expected = center(self.negative.clone())['center_heatmap'][0].numpy()
        torch.testing.assert_close(torch.tensor(rendered[0]), torch.tensor(expected), rtol=1e-5, atol=1e-6)

    def test_missing_running_statistics_are_rejected_instead_of_using_initial_values(self):
        values = {k:v for k,v in self.state['center_model_state_dict'].items()
                  if not k.endswith(('.running_mean', '.running_var'))}
        with self.assertRaisesRegex(ValueError, 'running statistics'):
            self.extras.load_center_model(self.args, {'center_model_state_dict':values}, 'cpu')

    def test_missing_learned_weights_are_rejected(self):
        values = dict(self.state['center_model_state_dict'])
        del values['module.instance_center_estimator.conv_end.0.weight']
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            self.extras.load_center_model(self.args, {'center_model_state_dict':values}, 'cpu')

    def test_nonfinite_running_statistics_are_rejected(self):
        values = dict(self.state['center_model_state_dict'])
        name = next(k for k in values if k.endswith('.running_var'))
        values[name] = torch.full_like(values[name], float('nan'))
        with self.assertRaisesRegex(ValueError, 'running statistics'):
            self.extras.load_center_model(self.args, {'center_model_state_dict':values}, 'cpu')


if __name__ == '__main__':
    unittest.main()
