"""Strict options, including a workaround for old Toolbox boolean parsing."""
from .stem_tasks import TaskConfig
from .semantic_classes import resolve_classes


def parse_bool(value, default):
    if value is None:
        return default
    if type(value) is bool:
        return value
    if value in ('true', 'false'):
        return value == 'true'
    raise ValueError('task option must be true or false')


def task_config(options, *, manifest=None, checkpoint=None):
    particles = parse_bool(options.get('nanoparticles'), True)
    semantic = parse_bool(options.get('segmentation'), False)
    classes = resolve_classes(options.get('semantic_classes'),
                              manifest=manifest if semantic else None,
                              checkpoint=checkpoint if semantic else None)
    return TaskConfig(particles, semantic, classes)
