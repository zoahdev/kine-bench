"""Adapter registry: map a CLI alias to a WorldModelAdapter instance."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import WorldModelAdapter
from .native import NativeAdapter
from .vjepa2 import MODELS as VJEPA2_MODELS, VJEPA2Adapter


def build_adapter(spec: str, ckpt: str | None = None, img_size: int = 224,
                  num_frames: int = 16, batch_size: int = 2) -> WorldModelAdapter:
    """`spec` is either a registry alias or `kineone-wm:<path-to-ckpt>`."""
    spec = (spec or "kineone-wm").strip()

    if spec in ("kineone-wm", "native", "kine-jepa"):
        return NativeAdapter(ckpt=ckpt, img_size=img_size, num_frames=num_frames)
    if spec.startswith("kineone-wm:") or spec.startswith("native:"):
        return NativeAdapter(ckpt=spec.split(":", 1)[1], img_size=img_size, num_frames=num_frames)

    if spec in VJEPA2_MODELS:
        return VJEPA2Adapter(alias=spec, num_frames=num_frames, batch_size=batch_size)

    raise KeyError(
        f"unknown model {spec!r}. available: {list_adapters()}"
    )


def list_adapters() -> List[str]:
    return ["kineone-wm"] + sorted(VJEPA2_MODELS)


def describe_all() -> Dict[str, Any]:
    out = {}
    for alias in list_adapters():
        if alias == "kineone-wm":
            out[alias] = {
                "source": "kine-jepa checkpoint (local)",
                "license": "MIT (in-house)",
                "capabilities": ["encode", "predict", "intervene?"],
                "params": "ViT-S/16 (22M)",
            }
        else:
            repo, lic, size = VJEPA2_MODELS[alias]
            out[alias] = {
                "source": f"hf:{repo}",
                "license": lic,
                "capabilities": ["encode"],
                "params": size,
            }
    return out
