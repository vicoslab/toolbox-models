"""Temporarily configure CeDiRNet's candidate extraction threshold."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


@contextmanager
def center_detection_threshold(center_model, threshold):
    """Apply a score threshold before center candidates and angles are built.

    CeDiRNet's peak detector uses a strict ``>`` comparison while validation
    uses ``>=``. Moving by one representable float preserves the public
    validation semantics at the boundary.
    """
    # None removes confidence filtering: keep every positive-response peak,
    # excluding the zero plateau created by the upstream ReLU (not detections).
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("center detection threshold must be finite and non-negative")

    model = _unwrap_model(center_model)
    localizer = model.instance_center_estimator
    previous = localizer.local_max_thr
    localizer.local_max_thr = (0.0 if threshold is None else float(
        np.nextafter(np.float32(threshold), np.float32(-np.inf))
    ))
    try:
        yield
    finally:
        localizer.local_max_thr = previous
