"""Locate kineworld_jepa and load KINE-JEPA checkpoints."""

import os
import sys
from pathlib import Path

import torch


def ensure_jepa_path() -> None:
    """Make `import kineworld_jepa` work: env var KINE_JEPA_ROOT or a sibling clone."""
    try:
        import kineworld_jepa  # noqa: F401
        return
    except ImportError:
        pass
    env = os.environ.get("KINE_JEPA_ROOT")
    here = Path(__file__).resolve().parent
    candidates = ([env] if env else []) + [
        str(here.parents[1] / "kine-jepa"),
        str(here.parents[1] / "kine-exp001"),
    ]
    for c in candidates:
        if c and (Path(c) / "kineworld_jepa").is_dir():
            sys.path.insert(0, c)
            return
    raise ImportError(
        "kineworld_jepa not found. Clone https://github.com/zoahdev/kine-jepa next to "
        "kine-bench, or set KINE_JEPA_ROOT to its directory."
    )


def load_model(ckpt_path, device, img_size=224, num_frames=16):
    """Build KineJEPA from checkpoint config and load weights. ckpt_path=None -> random init."""
    ensure_jepa_path()
    from kineworld_jepa.jepa import KineJEPA

    cfg, state = {}, None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"]
        cfg = ckpt.get("config") or {}
    model = KineJEPA(
        img_size=cfg.get("img_size", img_size),
        num_frames=cfg.get("num_frames", num_frames),
        enc_depth=cfg.get("enc_depth", 12),
        pred_depth=cfg.get("pred_depth", 6),
    )
    if state is not None:
        model.load_state_dict(state)
    model.to(device).eval()
    return model
