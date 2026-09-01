"""Adapter for KineOne-WM checkpoints (the in-house model)."""

from __future__ import annotations

from typing import Any

from .base import AdapterInfo, WorldModelAdapter


class NativeAdapter(WorldModelAdapter):
    """Loads a KineOne-WM-Latent checkpoint via kine-bench's own loader.

    Capabilities are resolved *after* loading, because a checkpoint only
    exposes `intervene` when it ships an InterventionHead (KINE-EXP-002 arm C).
    """

    def __init__(self, ckpt: str | None = None, img_size: int = 224, num_frames: int = 16):
        self.ckpt = ckpt
        self.img_size = img_size
        self.num_frames = num_frames
        self.info = AdapterInfo(
            name="kineone-wm-latent",
            source=ckpt or "random-init",
            license="MIT (kine-jepa, in-house)",
            capabilities=frozenset({"encode", "predict"}),
            params="ViT-S/16 (22M)",
            notes="In-house model. Gains 'intervene' only if the checkpoint carries an InterventionHead.",
        )

    def build(self, device: str = "cpu") -> Any:
        from ..load import load_model

        model = load_model(self.ckpt, device, img_size=self.img_size, num_frames=self.num_frames)
        model.to(device).eval()
        caps = {"encode", "predict"}
        if getattr(model, "intervention_head", None) is not None:
            caps.add("intervene")
        self.info.capabilities = frozenset(caps)
        self.info.name = "kineone-wm-latent" + ("+do" if "intervene" in caps else "")
        return model
