"""Self-contained synthetic clips so KINE-Bench runs without kine-jepa installed.

Distribution matches kineworld_jepa.dataset.SyntheticVideoDataset (moving
gradient ramp + 0.1 uniform noise - 0.5), with one difference that matters for
cross-model comparison: clips are **deterministic per index**. The original
sampler calls torch.rand() unseeded, so two runs -- or two models -- would see
different clips. Benchmarking models against each other on different data is
meaningless, so the noise here is drawn from a per-index generator.

Seed is also exposed so a third party can regenerate the exact same clips.
"""

from __future__ import annotations

import math

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def synthetic_clip(index: int, num_frames: int = 16, size: int = 224, seed: int = 0) -> torch.Tensor:
    """Return one (C, T, H, W) clip in KINE-Bench's normalized value range."""
    g = torch.Generator().manual_seed(seed + index * 7919)
    t = torch.linspace(0, 2 * math.pi, num_frames)
    phase = torch.rand(1, generator=g) * 2 * math.pi
    ramp = (torch.sin(t + phase) + 1) / 2                      # (T,)
    x = ramp.view(1, num_frames, 1, 1).expand(3, -1, size, size)
    noise = torch.rand(3, num_frames, size, size, generator=g) * 0.1
    clip = x + noise - 0.5
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return (clip - mean) / std


def synthetic_clips(n: int, num_frames: int = 16, size: int = 224, seed: int = 0) -> list:
    return [synthetic_clip(i, num_frames, size, seed) for i in range(n)]
