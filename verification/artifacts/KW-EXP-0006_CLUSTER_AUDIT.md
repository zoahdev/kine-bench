# KW-EXP-0006 episode-cluster robustness audit

- Date: 2026-09-02
- Status: **PASS AS POST-HOC ROBUSTNESS CHECK**
- Scope: statistical re-analysis of the frozen 96-episode run; no new model inference

## Why this audit was required

The pre-registered Fisher exact test used 480 horizon-level observations. Five observations share an episode, so the point-level p-value does not by itself account for within-episode dependence. This audit treats the episode as the resampling and permutation unit.

## Frozen definitions retained

- Spike: latent error greater than **4× the global median over all 576 points**.
- High interaction: `obj_disp` at or above the global 75th percentile among the 480 points where it is defined.
- No threshold or grouping rule was tuned after inspecting the cluster-aware result.

## Result

| Check | Result |
|---|---:|
| High-motion spike rate | 7.50% |
| Lower-motion spike rate | 1.39% |
| Difference | +6.11 percentage points |
| Episode-profile permutation p, one-sided | 0.00320 |
| Episode-cluster bootstrap 95% CI, difference | +1.67 to +10.73 percentage points |
| Spearman `obj_disp` vs log-error | 0.4198 |
| Episode-profile permutation p, two-sided Spearman | <0.00010 |
| Episode-cluster bootstrap 95% CI, Spearman | 0.2958 to 0.5346 |

The association survives an episode-cluster robustness check. This strengthens the narrow E1 statement that, in this run, larger object displacement is associated with larger latent error.

## Hard boundary

This is post-hoc re-analysis by KineWorld, not independent reproduction. It does not establish causality, cross-task generality, model superiority, or deployment utility. The physical-ratio conformal band remains rejected because it is 8–22× wider than the pooled baseline.

## Reproduce

```powershell
python verification/scripts/kw_exp_0006_cluster_audit.py <pusht-base-diagnostics-root> --permutations 10000 --bootstraps 5000
```

Machine result: `results/kw_exp_0006_cluster_audit.json`.
