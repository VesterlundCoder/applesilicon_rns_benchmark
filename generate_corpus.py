#!/usr/bin/env python3
"""Generate a deterministic 6F5 CMF matrix-chain corpus.

The matrices are taken from the corrected Ramanujan Dreams Matrix.walk
semantics in lib/cmf_walk_corrected.py. Each rational axis operator A is
converted to a primitive integer matrix A* = s A for a nonzero rational
scalar s. This preserves the projective CMF limit while giving every exact
backend the same integer 6x6 multiplication workload.

The default sparse_x1 workload is intentionally conservative: z=1/2,
shift=(1,...,1), direction=(1,0,...,0), N=1000. It gives one genuine 6F5
axis operator per outer walk step and keeps each individual operator within
signed int64, while the accumulated product grows to arbitrary precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List, Sequence

MAGIC = b"RDCMF6I1"
DIM = 6
NSHIFT = 11


def lcm(a: int, b: int) -> int:
    return abs(a // math.gcd(a, b) * b) if a and b else 0


def primitive_integer_matrix(M: Sequence[Sequence[Fraction]]) -> List[int]:
    den = 1
    for row in M:
        for x in row:
            den = lcm(den, x.denominator)
    vals = [int(x * den) for row in M for x in row]
    g = 0
    for x in vals:
        g = math.gcd(g, abs(x))
    if g > 1:
        vals = [x // g for x in vals]
    for x in vals:
        if x:
            if x < 0:
                vals = [-v for v in vals]
            break
    return vals


def matmul_int(A: Sequence[int], B: Sequence[int]) -> List[int]:
    C = [0] * (DIM * DIM)
    for i in range(DIM):
        for k in range(DIM):
            aik = A[i * DIM + k]
            if aik == 0:
                continue
            base_b = k * DIM
            base_c = i * DIM
            for j in range(DIM):
                C[base_c + j] += aik * B[base_b + j]
    return C


def matrix_digest(A: Sequence[int]) -> str:
    h = hashlib.sha256()
    for x in A:
        b = str(x).encode("ascii")
        h.update(struct.pack("<I", len(b)))
        h.update(b)
    return h.hexdigest()


def parse_vec(s: str, n: int) -> List[int]:
    v = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(v) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated integers, got {len(v)}")
    return v


def preset_dir(name: str) -> List[int]:
    if name == "sparse_x1":
        return [1] + [0] * 10
    if name == "x6":
        return [1] * 6 + [0] * 5
    if name == "dense_pos":
        return [1] * 11
    raise ValueError(name)


def load_oracle(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        from lib import cmf_walk_corrected as cmf  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Could not import {repo_root}/lib/cmf_walk_corrected.py. "
            "Run this from a checkout of RamanujanDreams-RNS-Mac or pass --repo-root."
        ) from exc
    return cmf


def generate_ops(cmf, shift: Sequence[int], dirv: Sequence[int], z_num: int, z_den: int,
                 outer_steps: int) -> List[List[int]]:
    z = Fraction(z_num, z_den)
    rank = cmf.rank_for_z(z)
    if rank != DIM:
        raise ValueError("This benchmark is fixed to rank 6. Use z != 1, e.g. z=1/2.")

    pos = cmf.start_pos(shift)
    zero, one = Fraction(0), Fraction(1)
    ops: List[List[int]] = []
    max_abs_dir = max(abs(int(d)) for d in dirv)

    for outer in range(outer_steps):
        for level in range(max_abs_dir, 0, -1):
            for axis in range(NSHIFT - 1, -1, -1):
                if abs(int(dirv[axis])) < level:
                    continue
                sign = 1 if int(dirv[axis]) > 0 else -1
                W = [[one if i == j else zero for j in range(DIM)] for i in range(DIM)]
                try:
                    cmf.apply_axis_step(W, pos, axis, sign, z, rank, zero, one)
                except cmf.WalkSingularity as exc:
                    raise RuntimeError(
                        f"trajectory became singular at outer_step={outer}, axis={axis}, "
                        f"status={exc.status}. Choose another workload/shift."
                    ) from exc
                ops.append(primitive_integer_matrix(W))
    return ops


def conservative_product_bound_bits(ops: Iterable[Sequence[int]]) -> int:
    # max-entry norm: ||AB||_max <= DIM * ||A||_max * ||B||_max
    log2_bound = 0.0
    for A in ops:
        m = max(abs(x) for x in A)
        if m == 0:
            continue
        log2_bound += math.log2(DIM) + math.log2(m)
    return int(math.ceil(log2_bound)) + 1


def write_binary(path: Path, ops: Sequence[Sequence[int]]) -> str:
    h = hashlib.sha256()
    with path.open("wb") as f:
        header = struct.pack("<8sII", MAGIC, DIM, len(ops))
        f.write(header)
        h.update(header)
        for A in ops:
            if len(A) != DIM * DIM:
                raise ValueError("bad matrix size")
            for x in A:
                if not (-(1 << 63) <= x < (1 << 63)):
                    raise OverflowError(
                        f"operator entry needs {abs(x).bit_length()} bits and does not fit int64. "
                        "Use sparse_x1, fewer outer steps, or extend the binary format."
                    )
                b = struct.pack("<q", x)
                f.write(b)
                h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
                    help="RamanujanDreams-RNS-Mac checkout root")
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark_corpus"))
    ap.add_argument("--preset", choices=["sparse_x1", "x6", "dense_pos"], default="sparse_x1")
    ap.add_argument("--outer-steps", type=int, default=1000)
    ap.add_argument("--shift", type=str, default=",".join(["1"] * 11))
    ap.add_argument("--dir", dest="dir_custom", type=str, default=None,
                    help="override preset with 11 comma-separated integers")
    ap.add_argument("--z-num", type=int, default=1)
    ap.add_argument("--z-den", type=int, default=2)
    ap.add_argument("--no-reference", action="store_true",
                    help="skip exact Python-int final product")
    args = ap.parse_args()

    if args.outer_steps <= 0:
        ap.error("--outer-steps must be positive")
    if args.z_den == 0:
        ap.error("--z-den must be nonzero")

    shift = parse_vec(args.shift, NSHIFT)
    dirv = parse_vec(args.dir_custom, NSHIFT) if args.dir_custom else preset_dir(args.preset)
    cmf = load_oracle(args.repo_root.resolve())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ops = generate_ops(cmf, shift, dirv, args.z_num, args.z_den, args.outer_steps)

    max_op_abs = max(abs(x) for A in ops for x in A)
    max_op_bits = max_op_abs.bit_length() if max_op_abs else 0
    corpus_path = args.out_dir / "operators_i64.bin"
    corpus_sha = write_binary(corpus_path, ops)

    meta = {
        "format": "RDCMF6I1",
        "description": "primitive projective integer 6x6 operators from corrected Ramanujan Dreams 6F5 Matrix.walk semantics",
        "dim": DIM,
        "operator_count": len(ops),
        "outer_steps": args.outer_steps,
        "operators_per_outer_step": len(ops) / args.outer_steps,
        "preset": args.preset,
        "shift": shift,
        "direction": dirv,
        "z_num": args.z_num,
        "z_den": args.z_den,
        "max_operator_abs": str(max_op_abs),
        "max_operator_bits": max_op_bits,
        "operator_exact_input_flags": {
            "float32_integer_exact": max_op_bits <= 24,
            "df64_nominal_48bit_integer_exact": max_op_bits <= 48,
            "float64_integer_exact": max_op_bits <= 53,
        },
        "conservative_product_bound_bits": conservative_product_bound_bits(ops),
        "corpus_sha256": corpus_sha,
        "source_oracle": "lib/cmf_walk_corrected.py",
    }

    if not args.no_reference:
        P = [1 if i == j else 0 for i in range(DIM) for j in range(DIM)]
        for A in ops:
            P = matmul_int(P, A)
        max_abs = max(abs(x) for x in P)
        meta["reference_max_abs_bits"] = max_abs.bit_length() if max_abs else 0
        meta["reference_sha256"] = matrix_digest(P)
        ref = {
            "dim": DIM,
            "operator_count": len(ops),
            "sha256": meta["reference_sha256"],
            "max_abs_bits": meta["reference_max_abs_bits"],
            "entries_row_major": [str(x) for x in P],
        }
        (args.out_dir / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")

    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
