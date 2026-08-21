#!/usr/bin/env python3
"""Bit-growth / crossover sweep for the Apple Silicon 6F5 benchmark.

Generates a single long sparse_x1 corpus, then at a set of checkpoints N records:
  - bits(P_N)     : bit-length of the largest entry of the exact product,
  - K(N)          : number of 30-bit primes needed for exact RNS capacity,
  - T_gmp, T_flint, T_rns_cpu : median chain time (single result) at that N.

The exact product bits are obtained in one incremental Python-int pass; each
backend is timed by calling bench_native on the first N operators (--ops N).
This produces the data for the bit-growth/crossover figure: it shows where RNS
becomes attractive as the integers grow.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_corpus as gc  # noqa: E402


def k_for_max_abs(max_abs: int) -> tuple[int, int]:
    """Count 30-bit primes (descending from 2^30) until product > 2*max_abs."""
    if max_abs == 0:
        return 1, 1
    target = 2 * max_abs + 1
    M, K = 1, 0
    x = (1 << 30) - 35
    if x % 2 == 0:
        x -= 1
    from run_all import is_prime32  # reuse the vetted Miller-Rabin
    while M < target:
        if is_prime32(x):
            M *= x
            K += 1
        x -= 2
        if x <= 1 << 29:
            raise RuntimeError("prime generator exhausted")
    return K, M.bit_length()


def median_time(native: Path, corpus: Path, backend: str, n: int, K: int,
                repeats: int, warmups: int) -> dict | None:
    argv = [native, corpus, "--backend", backend, "--ops", n,
            "--repeats", repeats, "--warmups", warmups, "--threads", "1"]
    if backend == "rns_cpu":
        argv += ["--k", K]
    p = subprocess.run(list(map(str, argv)), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    rows = [json.loads(x) for x in p.stdout.splitlines() if x.strip().startswith("{")]
    if p.returncode != 0 or not rows:
        sys.stderr.write(f"[warn] {backend} N={n} failed:\n{p.stdout}\n")
        return None
    return rows[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--checkpoints", default="50,100,200,300,500,700,1000,1400,2000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(int(x) for x in args.checkpoints.split(",") if x.strip())
    n_max = max(args.n_max, checkpoints[-1])

    cmf = gc.load_oracle(args.repo_root.resolve())
    shift = [1] * gc.NSHIFT
    dirv = gc.preset_dir("sparse_x1")
    print(f"generating {n_max} operators (sparse_x1) ...", flush=True)
    ops = gc.generate_ops(cmf, shift, dirv, 1, 2, n_max)
    corpus = args.out_dir / "operators_bitgrowth.bin"
    gc.write_binary(corpus, ops)

    # Incremental exact product; record bits + K at each checkpoint.
    native = args.build / "bench_native"
    cps = set(checkpoints)
    P = [1 if i == j else 0 for i in range(gc.DIM) for j in range(gc.DIM)]
    meta_by_n: dict[int, dict] = {}
    for n in range(1, n_max + 1):
        P = gc.matmul_int(P, ops[n - 1])
        if n in cps:
            max_abs = max(abs(x) for x in P)
            bits = max_abs.bit_length()
            K, mod_bits = k_for_max_abs(max_abs)
            meta_by_n[n] = {"N": n, "bits": bits, "K": K, "modulus_bits": mod_bits}
            print(f"  N={n}: bits={bits} K={K}", flush=True)

    rows = []
    for n in checkpoints:
        m = meta_by_n[n]
        row = dict(m)
        for backend in ("gmp", "flint", "rns_cpu"):
            r = median_time(native, corpus, backend, n, m["K"], args.repeats, args.warmups)
            if r:
                row[f"t_{backend}_s"] = r.get("median_s")
                if backend == "rns_cpu":
                    row["t_rns_exact_result_s"] = r.get("exact_result_median_s")
                    row["crt_host_median_s"] = r.get("crt_host_median_s")
        rows.append(row)
        print("row:", json.dumps(row), flush=True)

    out = args.out_dir / "bitgrowth.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
