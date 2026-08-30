"""KINE-Bench CLI: python -m kinebench run [--ckpt P] [--data-dir D] [--smoke]"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .load import ensure_jepa_path, load_model
from .metrics import ALL_TASKS


def build_parser():
    ap = argparse.ArgumentParser(prog="kinebench", description="KINE-Bench v0.1 evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run all tasks on a checkpoint")
    p.add_argument("--ckpt", type=str, default=None, help="KINE-JEPA checkpoint (.pt); omit for random init")
    p.add_argument("--data-dir", type=str, default=None, help="kine-datapipe clips directory")
    p.add_argument("--smoke", action="store_true", help="synthetic clips + random-init model")
    p.add_argument("--max-clips", type=int, default=48)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=str, default=None, help="write results JSON here")
    return ap


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
        "version": "0.1.0",
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
    results["wall_s"] = round(time.time() - t0, 1)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bench] results -> {args.out}")

    print("\n| task | score | baseline |")
    print("|---|---|---|")
    for name, r in results["tasks"].items():
        vals = list(r.values())
        print(f"| {name} | {vals[0]} | {vals[1]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
