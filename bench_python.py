#!/usr/bin/env python3
"""CPU baselines for the Apple Silicon 6F5 benchmark corpus."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import struct
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

MAGIC = b"RDCMF6I1"
DIM = 6


def load_corpus(path: Path) -> List[List[int]]:
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError("corpus too short")
    magic, dim, nops = struct.unpack_from("<8sII", data, 0)
    if magic != MAGIC or dim != DIM:
        raise ValueError(f"bad corpus header: magic={magic!r} dim={dim}")
    need = 16 + nops * DIM * DIM * 8
    if len(data) != need:
        raise ValueError(f"bad corpus length: got {len(data)}, expected {need}")
    vals = struct.unpack_from(f"<{nops * DIM * DIM}q", data, 16)
    return [list(vals[i * 36:(i + 1) * 36]) for i in range(nops)]


def matmul_py(A: Sequence[int], B: Sequence[int]) -> List[int]:
    C = [0] * 36
    for i in range(6):
        ci = 6 * i
        for k in range(6):
            aik = A[ci + k]
            if aik == 0:
                continue
            bk = 6 * k
            for j in range(6):
                C[ci + j] += aik * B[bk + j]
    return C


def run_python_int(ops: Sequence[Sequence[int]]) -> List[int]:
    P = [1 if i == j else 0 for i in range(6) for j in range(6)]
    for A in ops:
        P = matmul_py(P, A)
    return P


def digest_ints(A: Sequence[int]) -> str:
    h = hashlib.sha256()
    for x in A:
        b = str(int(x)).encode("ascii")
        h.update(struct.pack("<I", len(b)))
        h.update(b)
    return h.hexdigest()


def exact_projective_unit(A: Sequence[int]) -> List[float]:
    m = max(abs(int(x)) for x in A)
    if m == 0:
        return [0.0] * len(A)
    with localcontext() as ctx:
        ctx.prec = 80
        dm = Decimal(m)
        return [float(Decimal(int(x)) / dm) for x in A]


def approx_projective_unit(A: Sequence[float]) -> List[float]:
    m = max(abs(float(x)) for x in A)
    if m == 0.0:
        return [0.0] * len(A)
    return [float(x) / m for x in A]


def projective_error(approx: Sequence[float], exact: Sequence[int]) -> Dict[str, float]:
    a = approx_projective_unit(approx)
    e = exact_projective_unit(exact)
    max_abs = max(abs(x - y) for x, y in zip(a, e))
    rels = [abs(x - y) / abs(y) for x, y in zip(a, e) if abs(y) > 1e-14]
    return {
        "projective_max_abs_error": max_abs,
        "projective_max_rel_error_nonzero": max(rels) if rels else 0.0,
    }


def timed(fn: Callable[[], object], repeats: int, warmups: int) -> Tuple[object, Dict[str, float]]:
    for _ in range(warmups):
        fn()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    samples: List[float] = []
    result = None
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            result = fn()
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) * 1e-9)
    finally:
        if gc_was_enabled:
            gc.enable()
    med = statistics.median(samples)
    mad = statistics.median(abs(x - med) for x in samples)
    return result, {
        "median_s": med,
        "min_s": min(samples),
        "max_s": max(samples),
        "mad_s": mad,
        "repeats": repeats,
        "warmups": warmups,
    }


def backend_python_int(ops):
    return run_python_int(ops), 0.0, {"version": sys.version.split()[0]}


def backend_sympy(ops):
    import sympy as sp
    t0 = time.perf_counter()
    mats = [sp.Matrix(6, 6, A) for A in ops]
    setup = time.perf_counter() - t0

    def run():
        P = sp.eye(6)
        for A in mats:
            P = P * A
        return [int(P[i, j]) for i in range(6) for j in range(6)]
    return run, setup, {"version": sp.__version__}


def backend_flint(ops, threads: int):
    import flint
    from flint import fmpz_mat
    flint.ctx.threads = threads
    t0 = time.perf_counter()
    mats = [fmpz_mat(6, 6, A) for A in ops]
    setup = time.perf_counter() - t0

    def run():
        P = fmpz_mat(6, 6, [1 if i == j else 0 for i in range(6) for j in range(6)])
        for A in mats:
            P = P * A
        return [int(P[i, j]) for i in range(6) for j in range(6)]
    return run, setup, {"version": getattr(flint, "__version__", "unknown"), "threads": threads}


def backend_gmpy2(ops):
    import gmpy2
    mpz = gmpy2.mpz
    t0 = time.perf_counter()
    mats = [[mpz(x) for x in A] for A in ops]
    setup = time.perf_counter() - t0

    def mul(A, B):
        C = [mpz(0) for _ in range(36)]
        for i in range(6):
            ci = 6 * i
            for k in range(6):
                aik = A[ci + k]
                if not aik:
                    continue
                bk = 6 * k
                for j in range(6):
                    C[ci + j] += aik * B[bk + j]
        return C

    def run():
        P = [mpz(1 if i == j else 0) for i in range(6) for j in range(6)]
        for A in mats:
            P = mul(P, A)
        return [int(x) for x in P]
    return run, setup, {"version": gmpy2.version()}


def backend_numpy(ops):
    import numpy as np
    t0 = time.perf_counter()
    mats = [np.asarray(A, dtype=np.float64).reshape(6, 6) for A in ops]
    setup = time.perf_counter() - t0

    def run():
        P = np.eye(6, dtype=np.float64)
        exp_sum = 0
        for A in mats:
            P = P @ A
            mx = float(np.max(np.abs(P)))
            if mx > 0.0 and np.isfinite(mx):
                _, ex = np.frexp(mx)
                P = np.ldexp(P, -int(ex))
                exp_sum += int(ex)
        return P.reshape(-1).tolist(), exp_sum
    return run, setup, {"version": np.__version__, "normalization": "power-of-two after every operator"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--ops", type=int, default=0, help="prefix length; 0 means all")
    ap.add_argument("--backends", default="python_int,sympy,flint,gmpy2,numpy")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--flint-threads", type=int, default=1)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    all_ops = load_corpus(args.corpus)
    n = len(all_ops) if args.ops <= 0 else min(args.ops, len(all_ops))
    ops = all_ops[:n]
    exact_ref = run_python_int(ops)
    ref_digest = digest_ints(exact_ref)
    ref_bits = max(abs(x).bit_length() for x in exact_ref)

    rows = []
    for name in [x.strip() for x in args.backends.split(",") if x.strip()]:
        row = {
            "backend": name,
            "operator_count": n,
            "dim": 6,
            "reference_sha256": ref_digest,
            "reference_max_abs_bits": ref_bits,
        }
        try:
            if name == "python_int":
                fn = lambda: run_python_int(ops)
                setup_s = 0.0
                extra = {"version": sys.version.split()[0]}
                result, timing = timed(fn, args.repeats, args.warmups)
                row.update(timing)
                row["setup_s"] = setup_s
                row.update(extra)
                row["sha256"] = digest_ints(result)
                row["correct"] = row["sha256"] == ref_digest
            elif name == "sympy":
                fn, setup_s, extra = backend_sympy(ops)
                result, timing = timed(fn, args.repeats, args.warmups)
                row.update(timing); row["setup_s"] = setup_s; row.update(extra)
                row["sha256"] = digest_ints(result); row["correct"] = row["sha256"] == ref_digest
            elif name == "flint":
                fn, setup_s, extra = backend_flint(ops, args.flint_threads)
                result, timing = timed(fn, args.repeats, args.warmups)
                row.update(timing); row["setup_s"] = setup_s; row.update(extra)
                row["sha256"] = digest_ints(result); row["correct"] = row["sha256"] == ref_digest
            elif name == "gmpy2":
                fn, setup_s, extra = backend_gmpy2(ops)
                result, timing = timed(fn, args.repeats, args.warmups)
                row.update(timing); row["setup_s"] = setup_s; row.update(extra)
                row["sha256"] = digest_ints(result); row["correct"] = row["sha256"] == ref_digest
            elif name == "numpy":
                fn, setup_s, extra = backend_numpy(ops)
                result, timing = timed(fn, args.repeats, args.warmups)
                vals, exp_sum = result
                row.update(timing); row["setup_s"] = setup_s; row.update(extra)
                row["normalization_exp_sum"] = exp_sum
                row.update(projective_error(vals, exact_ref))
                row["correct"] = None
            else:
                raise ValueError(f"unknown backend {name}")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
