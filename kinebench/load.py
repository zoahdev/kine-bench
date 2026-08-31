"""Locate kineworld_jepa and load KINE-JEPA checkpoints."""

import os
import sys
from pathlib import Path

import torch


def ensure_jepa_path() -> None:
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


def unwrap_state(state):
    """Accept EXP-001 keys or EXP-002 CausalKineJEPA keys (base.* / head.*)."""
    if not state:
        return state
    if any(k.startswith("base.") for k in state):
        return {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
    if any(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_model(ckpt_path, device, img_size=224, num_frames=16):
    ensure_jepa_path()
    from kineworld_jepa.jepa import KineJEPA

    cfg, state = {}, None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = unwrap_state(ckpt.get("model", ckpt))
        cfg = ckpt.get("config") or {}
    kw = dict(
        img_size=int(cfg.get("img_size", img_size)),
        num_frames=int(cfg.get("num_frames", num_frames)),
        enc_depth=int(cfg.get("enc_depth", 12)),
        pred_depth=int(cfg.get("pred_depth", 6)),
    )
    if cfg.get("tiny") or kw["img_size"] <= 64:
        kw.update(tubelet_t=2, patch_size=16, enc_dim=64, enc_heads=4, pred_dim=64, pred_heads=4)
        if int(cfg.get("enc_depth", 12)) <= 2:
            kw["enc_depth"] = int(cfg.get("enc_depth", 1))
            kw["pred_depth"] = int(cfg.get("pred_depth", 2))
    model = KineJEPA(**kw)
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[load] unexpected={len(unexpected)} (ok if EXP-002 head keys stripped)")
        if missing:
            print(f"[load] missing={list(missing)[:8]}")
    model.to(device).eval()
    return model
