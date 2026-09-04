"""Verified runtime acquisition of the default localization checkpoint."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


LOCALIZATION_CHECKPOINT_URL = (
    "https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/"
    "localization_checkpoint.pth"
)
LOCALIZATION_CHECKPOINT_SHA256 = (
    "cffcfde184a22c03a67ecc741f3943d0325d4aabe812cb1787796f236403df84"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_localization_checkpoint(
    destination,
    *,
    url: str = LOCALIZATION_CHECKPOINT_URL,
    expected_sha256: str = LOCALIZATION_CHECKPOINT_SHA256,
) -> Path:
    """Return a verified checkpoint, downloading atomically when unavailable."""
    destination = Path(destination)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f"{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            print(f"Downloading default localization checkpoint from {url}", flush=True)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)

        actual_sha256 = _sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Localization checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary, destination)
        temporary = None
        print(f"Default localization checkpoint ready at {destination}", flush=True)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
