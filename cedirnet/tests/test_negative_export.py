"""Submitted empty annotations are negatives; absent submissions are unlabeled."""
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import yaml

MODEL_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cedirnet_export', MODEL_DIR / 'ls_adapter.py')
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def point_tag():
    return dict(type='keypoint', original_width=100, original_height=80,
                value={'x': 25, 'y': 50})


class NegativeExportTest(unittest.TestCase):
    def export(self, tags):
        return adapter.export([tags], '/unused', ['image.png'], False)

    def test_submitted_empty_annotation_exports_empty_points(self):
        self.assertEqual(self.export([]), {'points': []})

    def test_no_submission_remains_unlabeled(self):
        self.assertIsNone(adapter.export([], '/unused', ['image.png'], False))
        self.assertIsNone(adapter.export(None, '/unused', ['image.png'], False))

    def test_choices_are_ignored_and_do_not_require_dimensions(self):
        tag = dict(type='choices', from_name='split', to_name='image',
                   value={'choices': ['Train']})
        self.assertEqual(self.export([tag]), {'points': []})
        self.assertEqual(self.export([tag, point_tag()]), {'points': [(25, 40)]})

    def test_positive_points_and_vectors_are_unchanged(self):
        self.assertEqual(self.export([point_tag()]), {'points': [(25, 40)]})
        vector = dict(type='vector', original_width=100, original_height=80,
                      value={'vertices': [{'x': 25, 'y': 50}, {'x': 50, 'y': 75}]})
        self.assertEqual(self.export([vector]), {'points': [[25, 40, 50, 60]]})

    def test_template_has_no_new_negative_confirmation_control(self):
        config = yaml.safe_load((MODEL_DIR / 'config.yml').read_text())
        xml = ET.fromstring(config['config'].replace('@SMART_TOOLS@', ''))
        self.assertIsNone(xml.find(".//Choices[@name='object_negative']"))

    def test_manifest_help_documents_submitted_negatives_and_missing_labels(self):
        description = json.loads((MODEL_DIR / 'model.json').read_text())['properties']['manifest']['description']
        self.assertIn('points: []', description)
        self.assertIn('submitted', description.lower())
        self.assertIn('missing', description.lower())


if __name__ == '__main__':
    unittest.main()
