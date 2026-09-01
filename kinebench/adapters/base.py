"""KINE-Bench adapter interface (v0.5).

Any world model can be scored on KINE-Bench by implementing this interface.
The contract is deliberately small:

    grid        -> (t, h, w) token grid the model produces for one clip
    target(x)   -> (B, N, D) full token features  (frozen, no grad)
    encoder(x, visible_idx=idx) -> (B, V, D) features at visible positions
    predictor(visible, vis_idx, mask_idx) -> (B, M, D) imagined future tokens

Adapters declare which probes they can actually serve. A probe whose required
capability is missing is reported as ``n/a`` **with a reason** -- never as a
fabricated number. This matters because most public world models release an
encoder but no predictor, so FUT-1 / EMB-1 / CAU-1 are simply not applicable
to them; claiming otherwise would make cross-model tables meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple

# probe name -> capability required from the adapter
PROBE_CAPABILITY: Dict[str, str] = {
    "KINE-TEMP-1": "encode",
    "KINE-MOT-1": "encode",
    "KINE-EVT-1": "encode",
    "KINE-FUT-1": "predict",
    "KINE-EMB-1": "predict",
    # CAU-1's encoder branch only needs `encode`; the do(x) branch additionally
    # needs `intervene` and is reported separately as auc_do.
    "KINE-CAU-1": "encode",
}

# sub-metrics that silently degrade instead of failing the whole probe
OPTIONAL_PARTS: Dict[str, str] = {
    "KINE-CAU-1": "intervene",   # -> auc_do
}

CAPABILITY_NOTES = {
    "encode": "frozen encoder producing per-token features",
    "predict": "a predictor that can imagine masked future tokens",
    "intervene": "a do(x)-conditioned head distinguishing interventions",
}


class UnsupportedProbe(Exception):
    """Raised when a probe is attempted on a model that cannot serve it."""


def na_result(probe: str, capability: str) -> Dict[str, Any]:
    return {
        "status": "n/a",
        "reason": f"{probe} requires '{capability}' ({CAPABILITY_NOTES.get(capability, '')}), "
                  f"which this model does not expose",
    }


@dataclass
class AdapterInfo:
    name: str
    source: str          # where the weights come from
    license: str         # license of the *weights*, not of kine-bench
    capabilities: FrozenSet[str]
    params: str = "unknown"
    notes: str = ""


class WorldModelAdapter:
    """Base class. Subclasses build a facade exposing the KineJEPA-shaped
    interface that the existing probe modules already consume."""

    info: AdapterInfo

    # -- to be implemented -------------------------------------------------
    def build(self, device: str = "cpu") -> Any:
        """Return the facade object consumed by the probes."""
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    @property
    def capabilities(self) -> FrozenSet[str]:
        return self.info.capabilities

    def supports(self, probe: str) -> bool:
        return PROBE_CAPABILITY.get(probe, "encode") in self.capabilities

    def unavailable(self, probe: str) -> Dict[str, Any]:
        return na_result(probe, PROBE_CAPABILITY.get(probe, "encode"))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.info.name,
            "source": self.info.source,
            "license": self.info.license,
            "params": self.info.params,
            "capabilities": sorted(self.capabilities),
            "notes": self.info.notes,
        }


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def unnormalize(clip, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """KINE-Bench clips are ImageNet-normalized (C,T,H,W). Public video models
    run their own preprocessor, so hand them back raw-ish [0,1] frames."""
    import torch

    m = torch.tensor(mean, device=clip.device).view(-1, 1, 1, 1)
    s = torch.tensor(std, device=clip.device).view(-1, 1, 1, 1)
    return (clip * s + m).clamp(0.0, 1.0)


def as_uint8_numpy(clip):
    """(C,T,H,W) float [0,1] -> (T,H,W,C) uint8, the layout video processors want."""
    x = unnormalize(clip).clamp(0, 1)
    x = (x * 255.0).round().to(torch_uint8())
    return x.permute(1, 2, 3, 0).contiguous().cpu().numpy()


def torch_uint8():  # pragma: no cover - trivial
    import torch

    return torch.uint8
