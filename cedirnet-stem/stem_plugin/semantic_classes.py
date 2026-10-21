"""Class-ID identity from exported supervision and saved semantic checkpoints."""
import json
from pathlib import Path


DEFAULT_CLASSES = ('Carbon', 'Film', 'Vacuum')


def validate_classes(value, context):
    if (not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 255
            or any(not isinstance(name, str) or not name.strip() for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{context}: semantic classes must be 2..255 unique nonempty names')
    return tuple(value)


def manifest_classes(manifest):
    """Inspect the same training fallback and validation split as ToolboxDataset."""
    path = Path(manifest)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('version', 0) < 2:
        raise ValueError(f'{path}: semantic class resolution requires manifest version >= 2')
    training_split = 'train' if 'train' in data else 'data'
    expected = None
    for split in (training_split, 'val'):
        items = data.get(split, [])
        if not isinstance(items, list):
            raise ValueError(f'{path}: split {split!r} must be a list')
        for index, item in enumerate(items):
            context = f'{path}: split {split!r} item {index}'
            if not isinstance(item, dict):
                raise ValueError(f'{context}: expected a manifest item object')
            # Unannotated/particle-only rows cannot define segmentation classes.
            if 'semantic_mask' not in item:
                continue
            if 'semantic_classes' not in item and 'semantic_classes' not in data:
                raise ValueError(f'{context}: missing semantic_classes; re-export the dataset')
            classes = validate_classes(item.get('semantic_classes', data.get('semantic_classes')), context)
            if expected is None:
                expected = classes
            elif classes != expected:
                raise ValueError(f'{context}: semantic classes/order {list(classes)!r} '
                                 f'differ from training class order {list(expected)!r}')
        if split == training_split and expected is None:
            raise ValueError(f'{path}: no training semantic_classes found in split {split!r}; '
                             'annotate segmentation and re-export the dataset')
    return expected


def checkpoint_classes(state):
    if not isinstance(state, dict):
        raise ValueError('checkpoint must be an object with semantic class metadata')
    candidates = []
    saved = state.get('tasks')
    if saved is not None:
        if not isinstance(saved, dict):
            raise ValueError('checkpoint tasks must be an object')
        if saved.get('segmentation'):
            if saved.get('ignore_index', 255) != 255:
                raise ValueError('checkpoint semantic ignore_index must be 255')
            candidates.append(validate_classes(saved.get('classes'), 'checkpoint tasks'))
    metadata = state.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('checkpoint metadata must be an object')
    if 'class_names' in metadata:
        if metadata.get('ignore_index', 255) != 255:
            raise ValueError('checkpoint semantic ignore_index must be 255')
        candidates.append(validate_classes(metadata['class_names'], 'checkpoint metadata'))
    if candidates and any(classes != candidates[0] for classes in candidates[1:]):
        raise ValueError('checkpoint semantic classes/order disagree between metadata sources')
    return candidates[0] if candidates else None


def resolve_classes(selection, *, manifest=None, checkpoint=None):
    automatic = selection is None or (isinstance(selection, str) and selection in ('', 'auto'))
    configured = None
    if not automatic:
        if isinstance(selection, str):
            try:
                selection = json.loads(selection)
            except json.JSONDecodeError as error:
                raise ValueError('semantic_classes must be auto or a JSON list of class names') from error
        configured = validate_classes(selection, 'configured semantic_classes')
    if manifest is not None and checkpoint is not None:
        raise ValueError('resolve semantic classes from a manifest or checkpoint, not both')
    source = None
    if manifest is not None:
        source = manifest_classes(manifest)
    elif checkpoint is not None:
        source = checkpoint_classes(checkpoint)
        if source is None and automatic:
            raise ValueError('checkpoint has no semantic classes metadata; provide the original '
                             'ordered semantic_classes JSON list explicitly')
    if configured is not None and source is not None and configured != source:
        raise ValueError(f'configured semantic classes/order {list(configured)!r} do not match '
                         f'{"manifest" if manifest is not None else "checkpoint"} '
                         f'semantic classes/order {list(source)!r}; class IDs cannot be relabeled')
    # Preserve the host-neutral legacy task_config() API when no source is given.
    return configured if configured is not None else source if source is not None else DEFAULT_CLASSES
