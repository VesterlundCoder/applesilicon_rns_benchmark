#!/usr/bin/env python3
"""Benchmark the repository's actual corrected Metal 6F5 df64 walk.

This is deliberately separate from the dense 6x6 cross-backend kernel corpus.
It measures the real structured O(r^2) Ramanujan Dreams walk with unique,
deterministic canonical non-negative trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path


def make_trajectories(path: Path, count: int, dmax: int, seed: int) -> None:
    # Small local deterministic generator, no NumPy dependency.
    import random
    rng = random.Random(seed)
    shift = [1] * 11
    with path.open("wb") as f:
        for _ in range(count):
            a = sorted(rng.randint(0, dmax) for _ in range(6))
            b = sorted(rng.randint(0, dmax) for _ in range(5))
            d = a + b
            if not any(d):
                d[0] = 1
            f.write(struct.pack("<22i", *(shift + d)))


def run_once(binary: Path, traj: Path, out: Path, nsteps: int, chunk: int, cwd: Path = None) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(
        [str(binary), str(traj), str(out), "1", "2", str(nsteps), str(chunk)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        cwd=str(cwd) if cwd else None,
    )
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"production df64 failed ({p.returncode}):\n{p.stderr}")
    total = re.search(
        r"TOTAL\s+(\d+) trajectories \((\d+) OK, (\d+) flagged\) in ([0-9.eE+-]+) s wall "
        r"\(([0-9.eE+-]+) s gpu\) => ([0-9.eE+-]+) traj/s", p.stderr)
    row = {"external_wall_s": wall, "stderr": p.stderr}
    if total:
        row.update({
            "n_trajectories": int(total.group(1)),
            "n_ok": int(total.group(2)),
            "n_flagged": int(total.group(3)),
            "reported_wall_s": float(total.group(4)),
            "reported_gpu_s": float(total.group(5)),
            "reported_traj_per_s": float(total.group(6)),
        })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=None)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--dmax", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--nsteps", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--keep-output", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    binary = args.binary or (args.repo_root / "build" / "dreams_rns_cmf_df64")
    if not binary.exists():
        raise SystemExit(f"missing {binary}; build the repository first")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rd6f5_prod_") as td:
        td = Path(td)
        traj = td / "trajectories.bin"
        make_trajectories(traj, args.count, args.dmax, args.seed)
        rows = []
        for r in range(args.repeats):
            out = td / f"out_{r}.bin"
            row = run_once(binary, traj, out, args.nsteps, args.chunk, cwd=args.repo_root)
            row.update({
                "backend": "production_df64_gpu",
                "repeat": r,
                "z": "1/2",
                "rank": 6,
                "nsteps": args.nsteps,
                "direction_distribution": f"canonical sorted blocks, integer uniform [0,{args.dmax}]",
                "seed": args.seed,
                "timing_note": "reported_gpu_s is repository timing; for publication apply production_timing_patch.diff to use MTLCommandBuffer GPUStartTime/GPUEndTime",
            })
            rows.append(row)
            if args.keep_output:
                dst = args.output.parent / f"production_df64_repeat_{r}.bin"
                os.replace(out, dst)
            print(json.dumps({k: v for k, v in row.items() if k != "stderr"}, sort_keys=True))
        args.output.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
