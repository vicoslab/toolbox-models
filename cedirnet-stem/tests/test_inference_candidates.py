"""Exercise entrypoint prediction without loading a model at import time."""
import ast
from pathlib import Path


def test_browser_keeps_candidates_but_preannotations_keep_configured_cutoff():
    tree = ast.parse((Path(__file__).parents[1] / 'infer.py').read_text())
    predict = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
    calls = []
    class Runtime:
        def predict(self, images, **kwargs):
            calls.append(kwargs)
            return {'scores': [[.1, .8]]}
    ns = {'RUNTIME': Runtime(), 'CMD_ARGS': {'width': 512, 'height': 512, 'score_threshold': .5}}
    exec(compile(ast.Module(body=[predict], type_ignores=[]), 'infer.py', 'exec'), ns)
    ns['predict']([])
    assert calls[-1]['score_threshold'] == .5
    response = ns['predict']([], interactive=True)
    assert calls[-1]['score_threshold'] == 0
    assert response['candidate_score_threshold'] == 0
    assert response['display_score_threshold'] == .5


def test_infer_route_requests_interactive_candidates():
    tree = ast.parse((Path(__file__).parents[1] / 'infer.py').read_text())
    route = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'infer')
    calls = [n for n in ast.walk(route) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'predict']
    assert any(k.arg == 'interactive' and k.value.value is True for k in calls[0].keywords)
