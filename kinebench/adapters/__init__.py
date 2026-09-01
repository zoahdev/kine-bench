"""KINE-Bench adapters: score any world model, not just the in-house one.

    from kinebench.adapters import build_adapter
    adapter = build_adapter("vjepa2-vitl-256")
    model = adapter.build("cuda")
    adapter.supports("KINE-FUT-1")   # False -> report n/a, never fabricate
"""

from .base import (
    AdapterInfo,
    UnsupportedProbe,
    WorldModelAdapter,
    na_result,
    PROBE_CAPABILITY,
)
from .native import NativeAdapter
from .vjepa2 import VJEPA2Adapter, MODELS as VJEPA2_MODELS
from .registry import build_adapter, list_adapters, describe_all

__all__ = [
    "AdapterInfo",
    "UnsupportedProbe",
    "WorldModelAdapter",
    "na_result",
    "PROBE_CAPABILITY",
    "NativeAdapter",
    "VJEPA2Adapter",
    "VJEPA2_MODELS",
    "build_adapter",
    "list_adapters",
    "describe_all",
]
