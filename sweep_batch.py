#!/usr/bin/env python3
"""Batch-scaling sweep for the Apple Silicon 6F5 benchmark.

For a fixed corpus (default sparse_x1, N=1000) sweep the batch size B and record
throughput (trajectories or chains per second) for:

  - rns_gpu   : Metal RNS modular kernel throughput (residues only, exact-modular)
  - df64_gpu  : Metal double-float kernel throughput (approximate)
  - cpu_gmp   : trajectory-parallel GMP big-integer chains, end-to-end EXACT
  - cpu_accel : trajectory-parallel Accelerate dgemm, approximate
  - crt       : parallel centered-CRT reconstruction, the end-to-end EXACT
                throughput ceiling of the RNS path (kernel + reconstruction)

The GPU numbers use batch replay of the same operator chain to measure the
saturated kernel; they are labelled as such and are directly comparable to the
CPU trajectory-parallel throughput, which also replays the same chain.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_json(argv):
    p = subprocess.run(list(map(str, argv)), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    rows = [json.loads(x) for x in p.stdout.splitlines() if x.strip().startswith("{")]
    if p.returncode != 0 or not rows:
        sys.stderr.write(f"[warn] command failed: {' '.join(map(str, argv))}\n{p.stdout}\n")
        return None
    return rows[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True, help="build dir with bench_native/bench_metal")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True, help="RNS primes for exact capacity")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    native = args.build / "bench_native"
    metal = args.build / "bench_metal"
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows = []

    for B in batches:
        common = ["--repeats", args.repeats, "--warmups", args.warmups]

        r = run_json([metal, args.corpus, "--backend", "rns_gpu", "--batch", B,
                      "--k", args.k, *common])
        if r:
            rows.append({"family": "rns_gpu_kernel", "batch": B, "exact": "modular",
                         "throughput_per_s": r.get("trajectories_per_gpu_s"),
                         "gpu_median_s": r.get("gpu_median_s"), "raw": r})

        r = run_json([metal, args.corpus, "--backend", "df64_gpu", "--batch", B, *common])
        if r:
            rows.append({"family": "df64_gpu", "batch": B, "exact": "approx",
                         "throughput_per_s": r.get("trajectories_per_gpu_s"),
                         "gpu_median_s": r.get("gpu_median_s"), "raw": r})

        # CPU throughput needs batch>=2 for the parallel path; B=1 uses latency.
        if B >= 2:
            r = run_json([native, args.corpus, "--backend", "gmp", "--batch", B,
                          "--threads", args.threads, *common])
            if r:
                rows.append({"family": "cpu_gmp", "batch": B, "exact": "exact_end_to_end",
                             "throughput_per_s": r.get("throughput_chains_per_s"), "raw": r})

            r = run_json([native, args.corpus, "--backend", "accelerate", "--batch", B,
                          "--threads", args.threads, *common])
            if r:
                rows.append({"family": "cpu_accel", "batch": B, "exact": "approx",
                             "throughput_per_s": r.get("throughput_chains_per_s"), "raw": r})

            r = run_json([native, args.corpus, "--backend", "crt", "--batch", B,
                          "--threads", args.threads, "--k", args.k, *common])
            if r:
                rows.append({"family": "rns_exact_end_to_end_crt", "batch": B, "exact": "exact_end_to_end",
                             "throughput_per_s": r.get("throughput_chains_per_s"), "raw": r})

        print(f"batch {B}: " + ", ".join(
            f"{x['family']}={x['throughput_per_s']:.1f}/s" for x in rows if x["batch"] == B), flush=True)

    args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
