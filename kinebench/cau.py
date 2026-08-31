"""KINE-CAU-1: intervention discrimination on synthetic support-removal pairs."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def _clip(falling: bool, frames=16, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    video = torch.zeros(3, frames, size, size)
    video += 0.05 * torch.randn(3, frames, size, size, generator=g)
    bar_y = size // 2
    x0, w, h = size // 3, size // 8, size // 8
    for t in range(frames):
        if falling and t >= frames // 2:
            y = min(size - h - 1, bar_y - h + int((t - frames // 2) * (size / frames) * 1.8))
            draw_bar = False
        else:
            y = bar_y - h
            draw_bar = True
        video[:, t, y:y + h, x0:x0 + w] = 1.0
        if draw_bar:
            video[:, t, bar_y:bar_y + 3, size // 5: 4 * size // 5] = 0.7
    return video

def _feat(model, video, device):
    x = video.unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.target(x).mean(dim=1)
    return F.normalize(z, dim=-1)

def _auc(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1.0, len(scores) + 1.0, dtype=torch.float64)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = (ranks * labels).sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))

def intervention_auc(model, device, n_pairs=16, frames=16, size=64, seed=0):
    pos, neg = [], []
    pix_pos, pix_neg = [], []
    for i in range(n_pairs):
        ctrl = _clip(False, frames, size, seed + i)
        inter = _clip(True, frames, size, seed + i)
        mid = frames // 2
        def late(v):
            tail = v[:, mid:]
            reps = (frames + tail.shape[1] - 1) // tail.shape[1]
            return tail.repeat(1, reps, 1, 1)[:, :frames]
        f_ctrl = _feat(model, late(ctrl), device)
        f_int = _feat(model, late(inter), device)
        early = late(ctrl[:, :mid].repeat(1, 2, 1, 1)[:, :frames] if False else ctrl)
        # control shift: early vs late of the same unfallen clip
        f_early = _feat(model, ctrl, device)
        pos.append(1.0 - F.cosine_similarity(f_ctrl, f_int).item())
        neg.append(1.0 - F.cosine_similarity(f_ctrl, f_early).item())
        pix_pos.append((late(ctrl) - late(inter)).abs().mean().item())
        pix_neg.append(0.0)
    scores = pos + neg
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    return {
        "auc": round(_auc(scores, labels), 4),
        "baseline": 0.5,
        "pixel_auc": round(_auc(pix_pos + pix_neg, labels), 4),
        "mean_do_shift": round(float(sum(pos) / len(pos)), 4),
        "mean_control_shift": round(float(sum(neg) / len(neg)), 4),
        "n_pairs": n_pairs,
    }
