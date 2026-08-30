"""KINE-EVT-1 (v0.2): physical-event probe.

Does the frozen representation shift more across detected physical events
(collisions / drops / topples, mined by kine-datapipe's `events` command)
than across arbitrary same-video windows? Scored as AUC; pixel-space frame
diffing serves as the perception baseline.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor(frames, size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    import cv2

    out = []
    for f in frames:
        h, w = f.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        f = cv2.resize(f[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    x = torch.from_numpy(np.stack(out)).permute(3, 0, 1, 2).float() / 255.0  # (3, T, H, W)
    m = torch.tensor(mean).view(3, 1, 1, 1)
    s = torch.tensor(std).view(3, 1, 1, 1)
    return (x - m) / s


def _read_window(path, sampled_idx, total, max_frames, num_frames, size):
    """Read num_frames consecutive frames around a sampled index (events.json space)."""
    import cv2

    start = int(round(sampled_idx))
    idxs = [start * (total - 1) / (max_frames - 1) + k for k in range(num_frames)]
    cap = cv2.VideoCapture(str(path))
    frames = []
    for pos in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(pos, total - 1)))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(frame)
    cap.release()
    return _to_tensor(frames, size)


def _auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U), ties get average rank."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1.0, len(scores) + 1.0, dtype=torch.float64)
    # average ranks for ties
    vals, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(vals), dtype=torch.float64)
    sums.scatter_add_(0, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = (ranks * labels).sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def event_shift(model, raw_dir, events_path, device,
                num_frames=16, img_size=224, max_frames=300,
                controls_per_event=2, seed=0):
    """KINE-EVT-1: AUC separating event-window representation shifts from control shifts."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    raw_dir, events_path = Path(raw_dir), Path(events_path)
    report = json.loads(events_path.read_text(encoding="utf-8"))
    lo, hi = num_frames, max_frames - num_frames

    event_shifts, control_shifts = [], []
    pixel_event, pixel_control = [], []
    n_used_events = 0
    for name, entry in sorted(report.items()):
        path = raw_dir / name
        if not path.is_file():
            continue
        import cv2
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total < max_frames:
            continue
        for ev in entry.get("events", []):
            f = int(ev["frame"])
            if not (lo <= f <= hi):
                continue
            pre = _read_window(path, f - num_frames, total, max_frames, num_frames, img_size)
            post = _read_window(path, f, total, max_frames, num_frames, img_size)
            if pre is None or post is None:
                continue
            with torch.no_grad():
                fp = model.target(pre.unsqueeze(0).to(device)).mean(dim=1)
                fq = model.target(post.unsqueeze(0).to(device)).mean(dim=1)
            d = 1.0 - F.cosine_similarity(fp, fq, dim=-1).item()
            event_shifts.append(d)
            pixel_event.append((post - pre).abs().mean().item())
            n_used_events += 1
            for _ in range(controls_per_event):
                s = rng.randrange(lo, hi - num_frames + 1)
                wp = _read_window(path, s, total, max_frames, num_frames, img_size)
                wq = _read_window(path, s + num_frames, total, max_frames, num_frames, img_size)
                if wp is None or wq is None:
                    continue
                with torch.no_grad():
                    gp = model.target(wp.unsqueeze(0).to(device)).mean(dim=1)
                    gq = model.target(wq.unsqueeze(0).to(device)).mean(dim=1)
                control_shifts.append(1.0 - F.cosine_similarity(gp, gq, dim=-1).item())
                pixel_control.append((wq - wp).abs().mean().item())

    if len(event_shifts) < 4 or len(control_shifts) < 4:
        return {"auc": None, "baseline": 0.5,
                "error": f"too few usable windows (events={len(event_shifts)}, controls={len(control_shifts)})"}

    scores = event_shifts + control_shifts
    labels = [1.0] * len(event_shifts) + [0.0] * len(control_shifts)
    auc = _auc(scores, labels)
    pix_auc = _auc(pixel_event + pixel_control, labels)
    return {
        "auc": round(auc, 4),
        "baseline": 0.5,
        "pixel_baseline_auc": round(pix_auc, 4),
        "mean_event_shift": round(float(np.mean(event_shifts)), 4),
        "mean_control_shift": round(float(np.mean(control_shifts)), 4),
        "n_events_used": n_used_events,
        "n_controls": len(control_shifts),
    }
