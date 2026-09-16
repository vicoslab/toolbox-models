"""Strict options, including a workaround for old Toolbox boolean parsing."""
import json
from .stem_tasks import TaskConfig


def parse_bool(value, default):
    if value is None:
        return default
    if type(value) is bool:
        return value
    if value in ('true', 'false'):
        return value == 'true'
    raise ValueError('task option must be true or false')


def task_config(options):
    classes = options.get('semantic_classes') or '["Carbon", "Film", "Vacuum"]'
    if isinstance(classes, str):
        classes = json.loads(classes)
    if not isinstance(classes, (tuple, list)):
        raise ValueError('semantic_classes must be a JSON list')
    return TaskConfig(parse_bool(options.get('nanoparticles'), True),
                      parse_bool(options.get('segmentation'), False), tuple(classes))
