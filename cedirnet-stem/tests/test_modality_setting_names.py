"""Keep related modality controls together in the name-sorted Toolbox menu."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modality_setting_titles_share_prefix_and_sort_together():
    properties = json.loads((ROOT / 'model.json').read_text())['properties']
    expected = {
        'modality_dropout': 'Modality dropout - Enable',
        'bf_drop_probability': 'Modality dropout - BF drop probability',
        'haadf_drop_probability': 'Modality dropout - HAADF drop probability',
        'modality_dropout_seed': 'Modality dropout - Seed',
    }
    assert {key: properties[key]['title'] for key in expected} == expected
    sorted_keys = sorted(properties, key=lambda key: properties[key]['title'].casefold())
    positions = sorted(sorted_keys.index(key) for key in expected)
    assert positions == list(range(positions[0], positions[0] + len(expected)))
