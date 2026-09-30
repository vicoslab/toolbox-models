"""Negative direction supervision uses existing outputs, never a presence head.

Set CEDIRNET_SOURCE to an installed/patched CeDiRNet-3DoF root to also run
actual dataset, groundtruth and criterion integration checks.
"""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

MODEL_DIR = Path(__file__).resolve().parents[1]


def policy():
    # Compile the real inline function without importing MLflow/the Trainer.
    tree = ast.parse((MODEL_DIR / 'train.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'omit_negative_distance_loss']
    if len(functions) != 1:
        raise AssertionError('train.py must define omit_negative_distance_loss directly')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'train.py', 'exec'), namespace)
    return namespace['omit_negative_distance_loss']


class NegativeLossPolicyTest(unittest.TestCase):
    def losses(self):
        parts = [torch.tensor([i + .25, i + .5], requires_grad=True) for i in range(8)]
        parts[2] = parts[4] + parts[5] + parts[6] + parts[7]
        parts[0] = parts[1] + parts[2] + parts[3]
        return tuple(parts)

    def test_only_negative_distance_is_excluded_from_loss_and_diagnostics(self):
        original = self.losses()
        instance = torch.zeros(2, 1, 4, 4, dtype=torch.int16)
        instance[1, 0, 0, 0] = 1  # presence is not a coordinate sentinel
        targets = torch.zeros(2, 13, 1, 4, 4)
        targets[1, 2, 0, 0, 0] = 1
        result = policy()(original, {'instance': instance, 'centerdir_groundtruth': [targets]})
        for index in range(8):
            torch.testing.assert_close(result[index][1], original[index][1], rtol=0, atol=0)
        self.assertEqual(result[6][0].item(), 0)
        for index in (1, 3, 4, 5, 7):
            torch.testing.assert_close(result[index], original[index], rtol=0, atol=0)
        torch.testing.assert_close(result[2][0], original[4][0] + original[5][0] + original[7][0])
        torch.testing.assert_close(result[0][0], original[0][0] - original[6][0])
        self.assertEqual(original[6][0].item(), 6.25)  # no in-place edits

    def test_positive_losses_are_unchanged(self):
        original = self.losses()
        instance = torch.ones(2, 4, 4, dtype=torch.int16)
        targets = torch.zeros(2, 13, 1, 4, 4)
        targets[:, 2, 0, 0, 0] = 1
        result = policy()(original, {'instance': instance, 'centerdir_groundtruth': [targets]})
        for actual, expected in zip(result, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_policy_has_no_separate_module(self):
        self.assertFalse((MODEL_DIR / 'negative_supervision.py').exists())

    def test_trainer_applies_policy_before_metrics_and_backward(self):
        tree = ast.parse((MODEL_DIR / 'train.py').read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        train = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        calls = [n for n in ast.walk(train) if isinstance(n, ast.Call)]
        policy_calls = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'omit_negative_distance_loss']
        self.assertEqual(len(policy_calls), 1)
        metrics = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == '_updated_per_epoch_sample_metrics')
        backward = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'backward')
        self.assertLess(policy_calls[0].lineno, metrics.lineno)
        self.assertLess(policy_calls[0].lineno, backward.lineno)


@unittest.skipUnless(os.getenv('CEDIRNET_SOURCE'), 'set CEDIRNET_SOURCE for real upstream integration')
class NegativeTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ['CEDIRNET_SOURCE'])
        sys.path[:0] = [str(source / 'src'), str(MODEL_DIR)]
        from base_config import get_args
        from criterions import get_criterion
        from datasets import get_centerdir_dataset
        from models.center_groundtruth import CenterDirGroundtruth
        from utils.utils import variable_len_collate
        cls.get_args = staticmethod(get_args)
        cls.get_criterion = staticmethod(get_criterion)
        cls.get_dataset = staticmethod(get_centerdir_dataset)
        cls.groundtruth_type = CenterDirGroundtruth
        cls.collate = staticmethod(variable_len_collate)
        torch.set_num_threads(1)

    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        Image.new('RGB', (64, 64), (80, 100, 120)).save(root / 'negative.png')
        Image.new('RGB', (64, 64), (120, 100, 80)).save(root / 'positive.png')
        manifest = {'train': [
            {'image_path': 'negative.png', 'points': []},
            {'image_path': 'positive.png', 'points': [[32, 40, 40, 48]]},
            {'image_path': 'unreviewed-not-on-disk.png'},
        ]}
        self.manifest = root / 'manifest.json'
        self.manifest.write_text(json.dumps(manifest))

    def prepare(self, orientation=False, augment=False):
        args = self.get_args(64, 64, enable_3dof=orientation)
        opts = copy.deepcopy(args['train_dataset']['kwargs'])
        opts['manifest'] = str(self.manifest)
        if not augment:
            opts['transform'].transforms = opts['transform'].transforms[:2]
        dataset, groundtruth = self.get_dataset('', opts, args['train_dataset']['centerdir_gt_opts'])
        criterion = self.get_criterion(args['loss_type'], args['loss_opts'], None, None)
        return args, dataset, groundtruth, criterion

    def test_explicit_negative_is_kept_and_missing_labels_are_skipped(self):
        args, dataset, groundtruth, criterion = self.prepare()
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['instance'].count_nonzero().item(), 0)

    def test_empty_targets_remain_zero_under_actual_augmentations(self):
        args, dataset, groundtruth, criterion = self.prepare(augment=True)
        for _ in range(12):
            sample = groundtruth(self.collate([dataset[0]]), torch.arange(1).int())
            maps = self.groundtruth_type.parse_groundtruth_map(sample['centerdir_groundtruth'])
            for key in ('gt_sin_th', 'gt_cos_th', 'gt_R'):
                self.assertEqual(maps[key].count_nonzero().item(), 0)

    def test_real_losses_and_gradients_for_empty_mixed_and_positive_batches(self):
        for orientation in (False, True):
            args, dataset, groundtruth, criterion = self.prepare(orientation=orientation)
            for indices in ([0], [0, 0], [0, 0, 0, 0], [0, 1], [1, 0], [1, 1]):
                with self.subTest(orientation=orientation, indices=indices):
                    sample = groundtruth(self.collate([dataset[i] for i in indices]), torch.arange(len(indices)).int())
                    maps = self.groundtruth_type.parse_groundtruth_map(sample['centerdir_groundtruth'])
                    prediction = torch.full((len(indices), args['num_vector_fields'] + 1, 64, 64), .25, requires_grad=True)
                    raw = criterion(prediction, sample, centerdir_gt=sample['centerdir_groundtruth'],
                                    ignore_mask=sample['ignore'] > 0, **args['loss_w'])
                    # Prove the upstream criterion itself succeeds BEFORE the policy.
                    self.assertTrue(all(torch.isfinite(v).all() for v in raw))
                    self.assertEqual(raw[0].shape, (len(indices),))
                    raw_grad, = torch.autograd.grad(raw[0].sum(), prediction, retain_graph=True)
                    self.assertTrue(torch.isfinite(raw_grad).all())
                    result = policy()(raw, sample)
                    result[0].sum().backward()
                    self.assertTrue(all(torch.isfinite(v).all() for v in result))
                    self.assertTrue(torch.isfinite(prediction.grad).all())
                    for b, index in enumerate(indices):
                        if index == 0:
                            self.assertEqual(maps['gt_sin_th'][b].count_nonzero().item(), 0)
                            self.assertEqual(maps['gt_cos_th'][b].count_nonzero().item(), 0)
                            self.assertTrue((prediction.grad[b, :2] > 0).all())
                            self.assertEqual(prediction.grad[b, 2:].count_nonzero().item(), 0)
                            self.assertEqual(result[6][b].item(), 0)
                        else:
                            torch.testing.assert_close(prediction.grad[b], raw_grad[b], rtol=0, atol=0)
                            self.assertGreater(maps['gt_sin_th'][b].abs().sum().item(), 0)
                    diagnostics = criterion.get_loss_dict(result)
                    torch.testing.assert_close(diagnostics['losses_centerdir_total']['r'], result[6].sum())

    def test_border_positive_losses_and_gradients_are_unchanged(self):
        for point in ([5, 5], [0, 5], [5, 0], [63, 63]):
            data = json.loads(self.manifest.read_text())
            data['train'][1]['points'] = [point]
            self.manifest.write_text(json.dumps(data))
            for orientation in (False, True):
                with self.subTest(point=point, orientation=orientation):
                    args, dataset, groundtruth, criterion = self.prepare(orientation=orientation)
                    sample = groundtruth(self.collate([dataset[0], dataset[1]]), torch.arange(2).int())
                    prediction = torch.full((2, args['num_vector_fields'] + 1, 64, 64), .25, requires_grad=True)
                    raw = criterion(prediction, sample, centerdir_gt=sample['centerdir_groundtruth'],
                                    ignore_mask=sample['ignore'] > 0, **args['loss_w'])
                    baseline_grad, = torch.autograd.grad(raw[0].sum(), prediction, retain_graph=True)
                    result = policy()(raw, sample)
                    gradient, = torch.autograd.grad(result[0].sum(), prediction)
                    self.assertGreater(raw[6][1].item(), 0)
                    for actual, expected in zip(result, raw):
                        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                    torch.testing.assert_close(gradient[1], baseline_grad[1], rtol=0, atol=0)
                    self.assertEqual(gradient[0, 2].count_nonzero().item(), 0)

    def test_null_directions_have_zero_loss_despite_arbitrary_distance(self):
        args, dataset, groundtruth, criterion = self.prepare()
        sample = groundtruth(self.collate([dataset[0], dataset[0]]), torch.arange(2).int())
        prediction = torch.zeros((2, args['num_vector_fields'] + 1, 64, 64))
        prediction[:, 2] = 17
        prediction.requires_grad_()
        raw = criterion(prediction, sample, centerdir_gt=sample['centerdir_groundtruth'],
                        ignore_mask=sample['ignore'] > 0, **args['loss_w'])
        result = policy()(raw, sample)
        torch.testing.assert_close(result[0], torch.zeros(2), rtol=0, atol=0)
        result[0].sum().backward()
        self.assertEqual(prediction.grad.count_nonzero().item(), 0)

    def test_negative_gradient_moves_both_signs_toward_zero(self):
        args, dataset, groundtruth, criterion = self.prepare()
        sample = groundtruth(self.collate([dataset[0]]), torch.arange(1).int())
        prediction = torch.full((1, args['num_vector_fields'] + 1, 64, 64), .25)
        prediction[:, 1] = -.25
        prediction.requires_grad_()
        raw = criterion(prediction, sample, centerdir_gt=sample['centerdir_groundtruth'],
                        ignore_mask=sample['ignore'] > 0, **args['loss_w'])
        policy()(raw, sample)[0].sum().backward()
        self.assertTrue((prediction.grad[:, 0] > 0).all())
        self.assertTrue((prediction.grad[:, 1] < 0).all())

    def test_ignored_negative_pixels_have_no_gradient(self):
        args, dataset, groundtruth, criterion = self.prepare()
        for all_ignored in (False, True):
            with self.subTest(all_ignored=all_ignored):
                sample = groundtruth(self.collate([dataset[0]]), torch.arange(1).int())
                if all_ignored:
                    sample['ignore'].fill_(1)
                else:
                    sample['ignore'][:, :, :16] = 1
                prediction = torch.full((1, args['num_vector_fields'] + 1, 64, 64), .25, requires_grad=True)
                raw = criterion(prediction, sample, centerdir_gt=sample['centerdir_groundtruth'],
                                ignore_mask=sample['ignore'] > 0, **args['loss_w'])
                result = policy()(raw, sample)
                result[0].sum().backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertEqual(prediction.grad[:, :, :16].count_nonzero().item(), 0)
                if all_ignored:
                    self.assertEqual(prediction.grad.count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
