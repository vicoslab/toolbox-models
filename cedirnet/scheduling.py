"""Scheduling helpers for CeDiRNet training."""


def should_validate(epoch, total_epochs, interval):
    """Validate at every interval and always after the final epoch."""
    if interval < 1:
        raise ValueError("validation interval must be at least one epoch")
    return (epoch + 1) % interval == 0 or epoch + 1 == total_epochs
