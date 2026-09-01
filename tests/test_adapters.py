"""Adapter-layer tests. Pure CPU, no model downloads required."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinebench.adapters import (  # noqa: E402
    PROBE_CAPABILITY,
    VJEPA2_MODELS,
    build_adapter,
    describe_all,
    list_adapters,
    na_result,
)
from kinebench.synth import synthetic_clips  # noqa: E402

PASSED = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"{status} {name}" + (f"  {extra}" if extra else ""))
    if cond:
        PASSED.append(name)
    else:
        raise AssertionError(name)


def test_registry():
    for alias in list_adapters():
        a = build_adapter(alias, ckpt=None)
        assert a.info.name
        assert a.info.license
    check("test_registry_all_aliases_build", True, f"{len(list_adapters())} aliases")


def test_unknown_alias_rejected():
    try:
        build_adapter("definitely-not-a-model")
    except KeyError:
        check("test_unknown_alias_rejected", True)
        return
    raise AssertionError("unknown alias should raise KeyError")


def test_vjepa2_licenses_declared():
    """Every external model we ship must declare a license. Non-commercial
    weights must never enter this registry."""
    for alias, (repo, lic, size) in VJEPA2_MODELS.items():
        assert lic in ("MIT", "Apache-2.0"), f"{alias} license {lic} is not clearly commercial-safe"
    check("test_vjepa2_licenses_declared", True, str({k: v[1] for k, v in VJEPA2_MODELS.items()}))


def test_native_supports_predict():
    a = build_adapter("kineone-wm")
    assert a.supports("KINE-FUT-1")
    assert a.supports("KINE-TEMP-1")
    check("test_native_supports_predict", True)


def test_vjepa2_encoder_only_gating():
    """V-JEPA 2 releases an encoder only. The bench must report FUT-1/EMB-1 as
    n/a rather than inventing a number."""
    a = build_adapter("vjepa2-vitl-256")
    assert a.supports("KINE-TEMP-1") is True
    assert a.supports("KINE-MOT-1") is True
    assert a.supports("KINE-FUT-1") is False, "encoder-only model must not claim predict"
    assert a.supports("KINE-EMB-1") is False
    na = a.unavailable("KINE-FUT-1")
    assert na["status"] == "n/a" and "reason" in na
    assert "score" not in na, "an n/a result must never carry a score"
    check("test_vjepa2_encoder_only_gating", True, na["reason"][:60] + "...")


def test_na_result_carries_no_number():
    for probe, cap in PROBE_CAPABILITY.items():
        r = na_result(probe, cap)
        assert r["status"] == "n/a"
        assert not any(k in r for k in ("accuracy", "cosine", "auc", "pearson_r"))
    check("test_na_result_carries_no_number", True)


def test_describe_all_shape():
    d = describe_all()
    assert "kineone-wm" in d
    for alias, meta in d.items():
        assert {"source", "license", "capabilities", "params"} <= set(meta)
    check("test_describe_all_shape", True)


def test_synth_deterministic_across_calls():
    """Two models must be scored on identical clips, or comparison is void."""
    a = synthetic_clips(4, num_frames=8, size=64, seed=0)
    b = synthetic_clips(4, num_frames=8, size=64, seed=0)
    for x, y in zip(a, b):
        assert torch.allclose(x, y), "synthetic clips must be reproducible"
    c = synthetic_clips(4, num_frames=8, size=64, seed=1)
    assert not torch.allclose(a[0], c[0]), "different seed should give different clips"
    check("test_synth_deterministic_across_calls", True)


def test_synth_value_range_matches_real_pipeline():
    """Regression guard: kineworld_jepa's SyntheticVideoDataset forgot the
    ImageNet normalization that VideoClipDataset applies, so smoke clips lived
    in a different value range than real clips. This bench's synthetic clips
    must match the REAL pipeline."""
    a = synthetic_clips(2, num_frames=8, size=64, seed=0)[0]
    assert a.shape == (3, 8, 64, 64)
    # ImageNet-normalized inputs go well below -0.5
    assert a.min() < -1.0, f"expected normalized range, got min={a.min():.3f}"
    check("test_synth_value_range_matches_real_pipeline", True, f"min={a.min():.2f} max={a.max():.2f}")


def test_unnormalize_roundtrip():
    from kinebench.adapters.base import as_uint8_numpy, unnormalize

    clip = synthetic_clips(1, num_frames=8, size=64, seed=0)[0]
    back = unnormalize(clip)
    assert 0.0 <= back.min() and back.max() <= 1.0, (back.min(), back.max())
    arr = as_uint8_numpy(clip)
    assert arr.shape == (8, 64, 64, 3) and arr.dtype.name == "uint8"
    check("test_unnormalize_roundtrip", True, f"uint8 {(arr.min(), arr.max())}")


if __name__ == "__main__":
    for fn in [
        test_registry,
        test_unknown_alias_rejected,
        test_vjepa2_licenses_declared,
        test_native_supports_predict,
        test_vjepa2_encoder_only_gating,
        test_na_result_carries_no_number,
        test_describe_all_shape,
        test_synth_deterministic_across_calls,
        test_synth_value_range_matches_real_pipeline,
        test_unnormalize_roundtrip,
    ]:
        fn()
    print(f"\nall {len(PASSED)} tests passed")
