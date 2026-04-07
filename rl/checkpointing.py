from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import warnings

import torch


def safe_torch_load(path: str, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Safely load a trusted project checkpoint.

    Notes:
    - `torch.load` uses pickle; never load untrusted files.
    - Uses `weights_only=True` when supported by runtime torch.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    warnings.warn(
        "Loading checkpoints requires trust in file origin. "
        "Do not load untrusted checkpoint files.",
        RuntimeWarning,
        stacklevel=2,
    )

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)

    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format: expected a dict")
    return ckpt


def validate_checkpoint_schema(ckpt: Dict[str, Any], required_keys: Iterable[str]) -> None:
    keys = set(ckpt.keys())
    missing = [k for k in required_keys if k not in keys]
    if missing:
        raise ValueError(f"Invalid checkpoint format; missing keys: {missing}")
