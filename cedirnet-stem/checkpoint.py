"""Checkpoint compatibility helpers for the radius-only STEM adaptation."""

from __future__ import annotations


def safe_torch_load(path, *, map_location):
    """Load tensor checkpoints while allowing legacy NumPy scalar metadata only."""
    import numpy as np
    import torch

    numpy_core = np._core if hasattr(np, "_core") else np.core
    numpy_scalar = numpy_core.multiarray.scalar
    safe_globals = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.float64,
        np.int64,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location=map_location, weights_only=True)


def load_compatible_model_state(model, checkpoint):
    """Load checkpoint tensors whose names and shapes match ``model``.

    The upstream CeDiRNet-STEM model has an additional circularity output head.
    Filtering by key and shape allows its shared center-direction and radius
    weights to initialize this point+radius adaptation while ignoring that head.
    """
    checkpoint_state = checkpoint.get("model_state_dict")
    if not checkpoint_state:
        raise ValueError("checkpoint does not contain model_state_dict")

    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    if not compatible_state:
        raise ValueError("checkpoint contains no model tensors compatible with this model")

    model.load_state_dict(compatible_state, strict=False)
    return set(checkpoint_state) - set(compatible_state)
