"""Missing task labels are not negative labels, nor malformed labels."""
import json

import pytest

from stem_plugin.stem_tasks import TaskConfig
from stem_plugin.toolbox_dataset import ToolboxDataset

CLASSES = ['Carbon', 'Film', 'Vacuum']


def manifest(tmp_path, items, split='data'):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(version=4, semantic_classes=CLASSES, **{split: items})))
    return path


def item(**labels):
    return dict(images=['absent-BF.png', 'absent-HAADF.png'], **labels)


@pytest.mark.parametrize('particles,semantic,expected', [(True, False, 3), (False, True, 3), (True, True, 2)])
def test_mixed_tasks_and_explicit_negative(tmp_path, particles, semantic, expected):
    rows = [item(points=[[1, 2, 3]], semantic_mask='mask.png'),
            item(points=[], semantic_mask='mask.png'), item(points=[]),
            item(semantic_mask='mask.png'), {}]
    path = manifest(tmp_path, rows)
    with pytest.warns(UserWarning, match=f'kept {expected} of 5, skipped {5-expected}') as caught:
        ds = ToolboxDataset(path, TaskConfig(particles, semantic))
    assert len(ds) == expected
    assert str(path) in str(caught[0].message)
    assert ds.items[1]['points'] == []
    assert all('points' in row for row in ds.items) if particles else True
    # No image files exist: filtering and constructor must not read them.
    assert not (tmp_path / 'absent-BF.png').exists()


@pytest.mark.parametrize('particles,semantic', [(True, False), (False, True), (True, True)])
def test_export_shaped_metadata(tmp_path, particles, semantic):
    rows = [item(points=[[1, 2, 3]], semantic_mask='mask.png') for _ in range(3)]
    rows += [item() for _ in range(17)]
    with pytest.warns(UserWarning, match='kept 3 of 20, skipped 17'):
        ds = ToolboxDataset(manifest(tmp_path, rows), TaskConfig(particles, semantic))
    assert len(ds) == 3


@pytest.mark.parametrize('particles,semantic', [(True, False), (False, True), (True, True)])
def test_all_missing_and_optional_validation(tmp_path, particles, semantic):
    tasks = TaskConfig(particles, semantic)
    path = manifest(tmp_path, [item(), {}])
    with pytest.warns(UserWarning, match='skipped 2'):
        with pytest.raises(ValueError, match='No usable samples'):
            ToolboxDataset(path, tasks)
    path = manifest(tmp_path, [item(), {}], 'val')
    with pytest.warns(UserWarning, match="split 'val'.*kept 0 of 2"):
        assert len(ToolboxDataset(path, tasks, split='val', allow_empty=True)) == 0
    assert len(ToolboxDataset(manifest(tmp_path, [], 'val'), tasks, split='val', allow_empty=True)) == 0


@pytest.mark.parametrize('labels,match', [
    ({'points': None, 'semantic_mask': 'mask.png'}, 'points must be a list'),
    ({'points': [[1, 2, -1]], 'semantic_mask': 'mask.png'}, 'radius'),
    ({'points': [], 'semantic_mask': None}, 'semantic_mask'),
    ({'points': [], 'semantic_mask': ''}, 'semantic_mask'),
    ({'points': [], 'semantic_mask': 'mask.png', 'semantic_classes': CLASSES[::-1]}, 'class order'),
    # Partial labels are still validated before filtering.
    ({'semantic_mask': 'mask.png', 'semantic_classes': CLASSES[::-1]}, 'class order'),
    ({'points': None}, 'points must be a list'),
])
def test_malformed_supervision_is_not_filtered(tmp_path, labels, match):
    with pytest.raises(ValueError, match=match):
        ToolboxDataset(manifest(tmp_path, [item(**labels)]), TaskConfig(True, True))


def test_labeled_image_pair_is_still_required(tmp_path):
    with pytest.raises(ValueError, match='images'):
        ToolboxDataset(manifest(tmp_path, [{'points': []}]), TaskConfig(True, False))


def test_labeled_missing_image_still_fails_on_access(tmp_path):
    ds = ToolboxDataset(manifest(tmp_path, [item(points=[])]), TaskConfig(True, False))
    with pytest.raises(FileNotFoundError):
        ds[0]
