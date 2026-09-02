# KW-RISK-CARD-0001 verification package

This package exposes the evidence boundary for KineWorld's first Failure & Risk Card. It separates three different checks that must not be conflated.

## Level 1 — artifact integrity

No GPU or network is required:

```bash
python verification/scripts/verify_risk_card_v0.py
```

Expected output begins with `KW-RISK-CARD-0001 integrity: PASS`. This proves only that the local card and frozen source artifacts match their published hashes and elementary arithmetic is consistent.

## Level 2 — internal analysis regeneration

Regenerate the card from the frozen KW-EXP-0006 evidence:

```bash
python verification/scripts/build_risk_card_v0.py
python -m unittest tests.test_risk_card_v0 -v
```

This checks deterministic derivation. It still does not independently reproduce model inference.

## Level 3 — independent model rerun

An external evaluator must independently obtain the checkpoint identified by SHA-256 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`, reconstruct the frozen KW-EXP-0006 protocol, rerun all 96 episodes, retain raw logs, regenerate the analysis, document every deviation, and complete `verification/third_party/KW_RISK_CARD_ATTESTATION_TEMPLATE.md`.

Independent upstream downloads:

- JEPA-WM Push-T checkpoint: `https://dl.fbaipublicfiles.com/jepa-wms/pt_jepa-wm.pth.tar`
- Push-T dataset mirror documented by the local provenance audit: `https://osf.io/download/k2d8w/`
- Upstream source: `https://github.com/facebookresearch/jepa-wms`

The KineWorld frozen runner configuration is `external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep96_physint.yaml`. The exact instrumentation patch is `verification/patches/KW-EXP-0006_physical_instrumentation.patch`, applied to the base commit recorded in `verification/patches/KW-EXP-0006_base_commit.txt`. The external evaluator must record its own source commit and must not reuse KineWorld's generated diagnostics as a substitute for inference.

Only Level 3 performed and signed by an independent party can upgrade the evidence beyond KineWorld's E1 internal reproduction. Passing Level 1 or Level 2 must never be advertised as third-party validation.

## Frozen artifacts

- `results/KW_RISK_CARD_v0.json`
- `docs/product/KINEWORLD_RISK_CARD_v0.md`
- `verification/manifests/KW-RISK-CARD-v0_manifest.json`
- `verification/manifests/KW-EXP-0006_manifest.json`
- `results/kw_exp_0006_analysis.json`
- `verification/scripts/build_risk_card_v0.py`
- `verification/scripts/verify_risk_card_v0.py`
- `tests/test_risk_card_v0.py`
- `verification/third_party/KW_RISK_CARD_ATTESTATION_TEMPLATE.md`
- `results/kw_exp_0006_cluster_audit.json`
- `verification/scripts/kw_exp_0006_cluster_audit.py`
- `verification/artifacts/KW-EXP-0006_CLUSTER_AUDIT.md`
- `verification/patches/KW-EXP-0006_base_commit.txt`
- `verification/patches/KW-EXP-0006_physical_instrumentation.patch`
- `external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep96_physint.yaml`

The cluster audit is a post-hoc KineWorld robustness check with episode as the resampling unit. It is included to correct for within-episode dependence in the original point-level Fisher test; it is not Level 3 validation.
