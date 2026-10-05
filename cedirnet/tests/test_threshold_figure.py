"""Actual saved diagnostic includes the exact filtering threshold."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from matplotlib import pyplot as plt


class ThresholdFigureTest(unittest.TestCase):
    def test_above_maximum_threshold_note_does_not_round_down(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('boundary_diagnostics', root / 'diagnostics.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = np.float32(.3)
        threshold = float(np.nextafter(score, np.float32(np.inf)))
        fig = module.plot_training_diagnostics(image=np.zeros((3, 16, 24)),
                    centers=np.empty((0, 2)), scores=np.empty(0), angles=np.empty(0),
                    direction_output=np.ones((2, 16, 24)), localization_response=np.zeros((16, 24)),
                    detection_score_threshold=threshold)
        try:
            note = next(t.get_text() for a in fig.axes for t in a.texts if 'threshold' in t.get_text().lower())
            displayed = float(note.split('≥')[1].strip())
            self.assertGreater(displayed, float(score))
            self.assertEqual(np.float32(displayed), np.float32(threshold))
        finally:
            plt.close(fig)

    def test_threshold_note_is_inside_saved_figure_without_overflow(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('threshold_diagnostics', root / 'diagnostics.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fig = module.plot_training_diagnostics(image=np.zeros((3, 16, 24)),
                    centers=np.array([[8, 6]]), scores=np.array([.03]), angles=np.array([0]),
                    direction_output=np.ones((2, 16, 24)), localization_response=np.ones((16, 24))*.03,
                    detection_score_threshold=.03)
        try:
            fig.canvas.draw()
            texts = [t for axis in fig.axes for t in axis.texts if 'threshold' in t.get_text().lower()]
            self.assertEqual(len(texts), 1)
            self.assertIn('0.03', texts[0].get_text())
            note = texts[0].get_window_extent(fig.canvas.get_renderer())
            self.assertTrue(fig.bbox.contains(note.x0, note.y0))
            self.assertTrue(fig.bbox.contains(note.x1, note.y1))
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / 'figure.png'
                fig.savefig(image)
                self.assertGreater(image.stat().st_size, 0)
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
