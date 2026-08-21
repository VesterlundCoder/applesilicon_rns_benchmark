#!/usr/bin/env python3
"""Verify exact RNS dumps and floating-point dumps against the exact reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from decimal import Decimal, localcontext
from pathlib import Path
from typing import List, Sequence, Tuple

DIM = 6
E = 36


def digest_ints(vals: Sequence[int]) -> str:
    h = hashlib.sha256()
    for x in vals:
        b = str(int(x)).encode("ascii")
        h.update(struct.pack("<I", len(b)))
        h.update(b)
    return h.hexdigest()


def load_reference(path: Path) -> List[int]:
    obj = json.loads(path.read_text())
    vals = [int(x) for x in obj["entries_row_major"]]
    if len(vals) != E:
        raise ValueError("reference must contain 36 entries")
    return vals


def load_rns(path: Path) -> Tuple[List[int], List[List[int]]]:
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError("RNS dump too short")
    magic, K, dim = struct.unpack_from("<8sII", data, 0)
    if magic != b"RNSR6I01" or dim != DIM:
        raise ValueError(f"bad RNS dump header: {magic!r}, dim={dim}")
    off = 16
    need = off + 4 * K + 4 * K * E
    if len(data) != need:
        raise ValueError(f"bad RNS dump size: got {len(data)}, expected {need}")
    primes = list(struct.unpack_from(f"<{K}I", data, off))
    off += 4 * K
    flat = struct.unpack_from(f"<{K * E}I", data, off)
    residues = [list(flat[k * E:(k + 1) * E]) for k in range(K)]
    return primes, residues


def crt_centered(primes: Sequence[int], residues: Sequence[Sequence[int]]) -> Tuple[List[int], int]:
    # Incremental CRT; reconstruct all 36 entries in parallel.
    x = [0] * E
    M = 1
    for k, p in enumerate(primes):
        inv = pow(M % p, -1, p)
        rk = residues[k]
        for e in range(E):
            t = ((rk[e] - (x[e] % p)) % p) * inv % p
            x[e] += M * t
        M *= p
    half = M // 2
    x = [v - M if v > half else v for v in x]
    return x, M


def load_f64(path: Path) -> Tuple[List[float], int]:
    data = path.read_bytes()
    if len(data) != 8 + 4 + 8 + E * 8:
        raise ValueError("bad float64 dump size")
    magic, dim = struct.unpack_from("<8sI", data, 0)
    if magic != b"F64R6I01" or dim != DIM:
        raise ValueError(f"bad float64 header: {magic!r}, dim={dim}")
    exp_sum = struct.unpack_from("<q", data, 12)[0]
    vals = list(struct.unpack_from(f"<{E}d", data, 20))
    return vals, exp_sum


def exact_projective(ref: Sequence[int]) -> List[float]:
    mx = max(abs(x) for x in ref)
    if mx == 0:
        return [0.0] * E
    with localcontext() as ctx:
        ctx.prec = 100
        dmx = Decimal(mx)
        return [float(Decimal(x) / dmx) for x in ref]


def float_projective(vals: Sequence[float]) -> List[float]:
    mx = max(abs(x) for x in vals)
    if mx == 0.0:
        return [0.0] * E
    return [x / mx for x in vals]


def verify_int_text(dump: Path, ref: Sequence[int]) -> dict:
    vals = [int(x.strip()) for x in dump.read_text().splitlines() if x.strip()]
    if len(vals) != E:
        raise ValueError(f"{dump}: expected 36 integer lines, got {len(vals)}")
    return {
        "kind": "exact_integer",
        "dump": str(dump),
        "correct": vals == list(ref),
        "sha256": digest_ints(vals),
        "reference_sha256": digest_ints(ref),
    }


def verify_rns(dump: Path, ref: Sequence[int]) -> dict:
    primes, residues = load_rns(dump)
    got, modulus = crt_centered(primes, residues)
    ref_max = max(abs(x) for x in ref)
    enough = modulus > 2 * ref_max
    correct = enough and got == list(ref)
    return {
        "kind": "rns",
        "dump": str(dump),
        "K": len(primes),
        "modulus_bits": modulus.bit_length(),
        "required_signed_bits": ref_max.bit_length() + 1,
        "capacity_sufficient": enough,
        "correct": correct,
        "sha256": digest_ints(got),
        "reference_sha256": digest_ints(ref),
    }


def verify_f64(dump: Path, ref: Sequence[int]) -> dict:
    vals, exp_sum = load_f64(dump)
    a = float_projective(vals)
    e = exact_projective(ref)
    abs_err = [abs(x - y) for x, y in zip(a, e)]
    rel_err = [abs(x - y) / abs(y) for x, y in zip(a, e) if abs(y) > 1e-15]
    return {
        "kind": "floating",
        "dump": str(dump),
        "normalization_exp_sum": exp_sum,
        "all_finite": all(math.isfinite(x) for x in vals),
        "projective_max_abs_error": max(abs_err),
        "projective_max_rel_error_nonzero": max(rel_err) if rel_err else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--int-text", type=Path, action="append", default=[])
    ap.add_argument("--rns", type=Path, action="append", default=[])
    ap.add_argument("--f64", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    ref = load_reference(args.reference)
    rows = ([verify_int_text(p, ref) for p in args.int_text] +
            [verify_rns(p, ref) for p in args.rns] +
            [verify_f64(p, ref) for p in args.f64])
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return 1 if any(r.get("kind") in {"rns", "exact_integer"} and not r.get("correct") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
