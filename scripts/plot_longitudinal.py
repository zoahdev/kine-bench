"""Plot longitudinal KINE-Bench curves from results/*-v03.json checkpoints."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TASKS = [
    ("KINE-TEMP-1", "accuracy", "baseline", "acc"),
    ("KINE-MOT-1", "pearson_r", "baseline", "r"),
    ("KINE-EVT-1", "auc", "baseline", "AUC"),
    ("KINE-FUT-1", "cosine", "random_baseline", "cosine"),
    ("KINE-EMB-1", "cosine", "random_baseline", "cosine"),
]


def load_series():
    series = {}
    for path in sorted(RESULTS_DIR.glob("*-v03.json")):
        m = re.search(r"step(\d+)", path.name)
        if not m:
            continue
        step = int(m.group(1))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        series[step] = data.get("tasks", {})
    return series


def main():
    series = load_series()
    if not series:
        raise SystemExit("no results/*-v03.json found")
    steps = sorted(series)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for task, score_key, base_key, unit in TASKS:
        xs, ys, bs = [], [], []
        for step in steps:
            t = series[step].get(task)
            if not t or score_key not in t:
                continue
            xs.append(step)
            ys.append(t[score_key])
            bs.append(t.get(base_key, t.get("random_baseline")))
        label = task.replace("KINE-", "").replace("-1", "")
        ax.plot(xs, ys, marker="o", label=f"{label} ({unit})")
        ax.plot(xs, bs, marker="x", linestyle="--", alpha=0.55)

    ax.set_xlabel("training steps")
    ax.set_ylabel("score")
    ax.set_title("KINE-Bench longitudinal curves (solid = score, dashed = baseline)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = RESULTS_DIR / "longitudinal.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot] wrote {out} from checkpoints {steps}")


if __name__ == "__main__":
    main()
