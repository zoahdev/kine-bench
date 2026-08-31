"""KINE-Bench CLI: python -m kinebench run [--ckpt P] [--data-dir D] [--smoke]"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .cau import intervention_auc
from .emb import embodied_imagination
from .events import event_shift
from .load import ensure_jepa_path, load_model
from .metrics import ALL_TASKS


def build_parser():
    ap = argparse.ArgumentParser(prog="kinebench", description="KINE-Bench v0.4 evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run all tasks on a checkpoint")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-clips", type=int, default=48)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--events-json", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return ap


def resolve_event_inputs(args):
    if args.raw_dir and args.events_json:
        return Path(args.raw_dir), Path(args.events_json)
    if args.data_dir:
        base = Path(args.data_dir)
        for c in (base, base.parent):
            if (c / "events.json").is_file() and (c / "raw").is_dir():
                return c / "raw", c / "events.json"
    return None


def load_clips(args):
    ensure_jepa_path()
    if args.smoke or not args.data_dir:
        from kineworld_jepa.dataset import SyntheticVideoDataset
        ds = SyntheticVideoDataset(num_frames=args.num_frames, size=args.img_size)
        n = min(len(ds), args.max_clips)
        clips = [ds[i] for i in range(n)]
        return clips, f"synthetic({n})"
    from kineworld_jepa.dataset import VideoClipDataset
    ds = VideoClipDataset(args.data_dir, num_frames=args.num_frames, size=args.img_size)
    n = min(len(ds), args.max_clips)
    clips = [ds[i] for i in range(n)]
    return clips, f"{args.data_dir} ({n}/{len(ds)} clips)"


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd != "run":
        return 1

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    clips, source = load_clips(args)
    print(f"[bench] data: {source} | device: {device}")
    model = load_model(args.ckpt, device, img_size=args.img_size, num_frames=args.num_frames)
    print(f"[bench] model: {args.ckpt or 'random-init'} | grid {tuple(model.grid)}")

    results = {
        "harness": "kinebench",
        "version": "0.4.0",
        "checkpoint": args.ckpt,
        "data": source,
        "num_clips": len(clips),
        "device": str(device),
        "tasks": {},
    }
    for name, fn in ALL_TASKS:
        r = fn(model, clips, device)
        results["tasks"][name] = r
        print(f"[bench] {name}: {r}")

    evt = resolve_event_inputs(args)
    if evt is not None:
        raw_dir, events_json = evt
        print(f"[bench] KINE-EVT-1 inputs: {raw_dir} + {events_json.name}")
        r = event_shift(model, raw_dir, events_json, device,
                        num_frames=args.num_frames, img_size=args.img_size)
        results["tasks"]["KINE-EVT-1"] = r
        print(f"[bench] KINE-EVT-1: {r}")
    else:
        print("[bench] KINE-EVT-1 skipped (no raw videos + events.json found)")

    r = embodied_imagination(model, device, num_frames=args.num_frames, img_size=args.img_size)
    results["tasks"]["KINE-EMB-1"] = r
    if r.get("error"):
        print(f"[bench] KINE-EMB-1 skipped ({r['error']})")
    else:
        print(f"[bench] KINE-EMB-1: {r}")

    cau_size = 64 if args.smoke or args.img_size < 128 else min(args.img_size, 128)
    r = intervention_auc(model, device, n_pairs=16, frames=min(args.num_frames, 16), size=cau_size)
    results["tasks"]["KINE-CAU-1"] = r
    print(f"[bench] KINE-CAU-1: {r}")

    results["wall_s"] = round(time.time() - t0, 1)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bench] results -> {args.out}")

    print("\n| task | score | baseline |")
    print("|---|---|---|")
    for name, r in results["tasks"].items():
        vals = list(r.values())
        score = vals[0] if vals[0] is not None else "n/a"
        print(f"| {name} | {score} | {vals[1] if len(vals) > 1 else '' } |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
