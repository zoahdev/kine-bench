"""Cluster-aware robustness audit for KW-EXP-0006.

The original point-level Fisher test does not model dependence among the five
usable horizon points from each episode.  This audit therefore treats an
episode as the resampling/permutation unit.  It is a post-hoc robustness check,
not a replacement for an independently replicated, pre-registered experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


SEED = 20260902


def load_rows(root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    all_error_rows: list[dict] = []
    for path in sorted(root.glob("ep_*/diagnostics.json")):
        episode = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for plan in payload:
            for point in plan.get("per_horizon_latent_mse", []):
                physical = point.get("physical") or {}
                all_error_rows.append(
                    {
                        "episode": episode,
                        "k": int(point["latent_step"]),
                        "error": float(point["predicted_actual_latent_mse"]),
                    }
                )
                if physical.get("obj_disp") is None:
                    continue
                rows.append(
                    {
                        "episode": episode,
                        "k": int(point["latent_step"]),
                        "error": float(point["predicted_actual_latent_mse"]),
                        "obj_disp": float(physical["obj_disp"]),
                    }
                )
    if not rows:
        raise SystemExit(f"No usable diagnostics found under {root}")
    return rows, all_error_rows


def statistics(rows: list[dict], spike_threshold: float, displacement_cutoff: float) -> dict:
    high = [r for r in rows if r["obj_disp"] >= displacement_cutoff]
    low = [r for r in rows if r["obj_disp"] < displacement_cutoff]
    high_rate = np.mean([r["error"] > spike_threshold for r in high])
    low_rate = np.mean([r["error"] > spike_threshold for r in low])
    rho = spearmanr(
        [r["obj_disp"] for r in rows],
        [np.log(max(r["error"], np.finfo(float).tiny)) for r in rows],
    ).statistic
    return {
        "high_spike_rate": float(high_rate),
        "low_spike_rate": float(low_rate),
        "spike_rate_difference": float(high_rate - low_rate),
        "relative_risk": float(high_rate / low_rate) if low_rate else None,
        "spearman_obj_disp_vs_log_error": float(rho),
    }


def permute_episode_profiles(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Permute complete error profiles across episodes, retaining each k slot."""
    episodes = sorted({r["episode"] for r in rows})
    shuffled = rng.permutation(episodes)
    source_for_target = dict(zip(episodes, shuffled))
    error_by_episode_k = {(r["episode"], r["k"]): r["error"] for r in rows}
    return [
        {**r, "error": error_by_episode_k[(source_for_target[r["episode"]], r["k"])]}
        for r in rows
    ]


def bootstrap_episodes(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    episodes = sorted({r["episode"] for r in rows})
    by_episode = {ep: [r for r in rows if r["episode"] == ep] for ep in episodes}
    sampled = rng.choice(episodes, size=len(episodes), replace=True)
    out: list[dict] = []
    for draw, ep in enumerate(sampled):
        out.extend({**r, "episode": f"draw_{draw}"} for r in by_episode[ep])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_root", type=Path)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--bootstraps", type=int, default=10000)
    args = parser.parse_args()

    rows, all_error_rows = load_rows(args.diagnostics_root)
    all_errors = np.asarray([r["error"] for r in all_error_rows])
    spike_threshold = 4.0 * float(np.median(all_errors))
    displacement_cutoff = float(np.percentile([r["obj_disp"] for r in rows], 75))
    observed = statistics(rows, spike_threshold, displacement_cutoff)
    rng = np.random.default_rng(SEED)

    null_diffs = np.empty(args.permutations)
    null_rhos = np.empty(args.permutations)
    for i in range(args.permutations):
        value = statistics(
            permute_episode_profiles(rows, rng), spike_threshold, displacement_cutoff
        )
        null_diffs[i] = value["spike_rate_difference"]
        null_rhos[i] = value["spearman_obj_disp_vs_log_error"]

    boot_diffs = np.empty(args.bootstraps)
    boot_rhos = np.empty(args.bootstraps)
    for i in range(args.bootstraps):
        sample = bootstrap_episodes(rows, rng)
        # Re-estimate both data-derived cutoffs inside each cluster bootstrap.
        sample_spike = 4.0 * float(np.median([r["error"] for r in sample]))
        sample_disp = float(np.percentile([r["obj_disp"] for r in sample], 75))
        value = statistics(sample, sample_spike, sample_disp)
        boot_diffs[i] = value["spike_rate_difference"]
        boot_rhos[i] = value["spearman_obj_disp_vs_log_error"]

    result = {
        "audit_id": "KW-EXP-0006-CLUSTER-AUDIT",
        "status": "POST_HOC_ROBUSTNESS_CHECK",
        "seed": SEED,
        "episode_is_resampling_unit": True,
        "episodes": len({r["episode"] for r in rows}),
        "points": len(rows),
        "points_used_for_global_spike_threshold": len(all_error_rows),
        "spike_definition": "error > 4 * global median error",
        "spike_threshold": spike_threshold,
        "obj_disp_high_definition": "obj_disp >= global 75th percentile",
        "obj_disp_cutoff": displacement_cutoff,
        "observed": observed,
        "episode_profile_permutation": {
            "n": args.permutations,
            "p_one_sided_spike_rate_difference": float(
                (1 + np.sum(null_diffs >= observed["spike_rate_difference"]))
                / (args.permutations + 1)
            ),
            "p_two_sided_spearman": float(
                (1 + np.sum(np.abs(null_rhos) >= abs(observed["spearman_obj_disp_vs_log_error"])))
                / (args.permutations + 1)
            ),
        },
        "episode_cluster_bootstrap_95pct": {
            "n": args.bootstraps,
            "spike_rate_difference": np.percentile(boot_diffs, [2.5, 97.5]).tolist(),
            "spearman": np.percentile(boot_rhos, [2.5, 97.5]).tolist(),
        },
        "claim_boundary": (
            "Robustness to episode clustering is not independent replication, "
            "does not establish causality, and does not validate deployment utility."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
