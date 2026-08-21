#!/usr/bin/env python3
"""Build and run the reproducible Apple Silicon 6F5 benchmark suite.

Run the same command on the M1 and M5 machines with different --label values.
Results are JSON/JSONL plus exact dump files that can be independently checked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(argv, *, cwd=None, log=None, allow_fail=False):
    print("+", " ".join(map(str, argv)), flush=True)
    p = subprocess.run(list(map(str, argv)), cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(p.stdout)
    print(p.stdout, end="")
    if p.returncode and not allow_fail:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, argv))}")
    return p


def is_prime32(n: int) -> bool:
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in [2, 3, 5, 7, 11]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def exact_k_for_reference(reference: Path) -> tuple[int, int]:
    obj = json.loads(reference.read_text())
    mx = max(abs(int(x)) for x in obj["entries_row_major"])
    target = 2 * mx + 1
    M, K = 1, 0
    x = (1 << 30) - 35
    if x % 2 == 0:
        x -= 1
    while M < target:
        if is_prime32(x):
            M *= x; K += 1
        x -= 2
        if x <= 1 << 29:
            raise RuntimeError("prime generator exhausted")
    return K, M.bit_length()


def parse_batches(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True,
                    help="checkout of VesterlundCoder/RamanujanDreams-RNS-Mac")
    ap.add_argument("--label", required=True, help="e.g. M1 or M5-Max")
    ap.add_argument("--workdir", type=Path, default=Path("apple6f5_results"))
    ap.add_argument("--preset", choices=["sparse_x1", "x6", "dense_pos"], default="sparse_x1")
    ap.add_argument("--outer-steps", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--gpu-batches", default="auto",
                    help="comma list, or auto: batch=1 plus a saturation batch targeting ~8192 prime lanes")
    ap.add_argument("--skip-python-slow", action="store_true",
                    help="skip SymPy/Python-int/gmpy2 full timings after reference generation")
    ap.add_argument("--skip-production", action="store_true")
    ap.add_argument("--production-count", type=int, default=20000)
    ap.add_argument("--quick", action="store_true", help="50 operators, 2 repeats, small GPU batches")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    work = args.workdir.resolve() / args.label
    corpus_dir = work / "corpus"
    logs = work / "logs"
    dumps = work / "dumps"
    for d in (corpus_dir, logs, dumps): d.mkdir(parents=True, exist_ok=True)

    outer = 50 if args.quick else args.outer_steps
    repeats = 2 if args.quick else args.repeats
    warmups = 0 if args.quick else args.warmups
    batches = None

    run([sys.executable, HERE / "collect_metadata.py", "--repo-root", repo,
         "--label", args.label, "--output", work / "system_metadata.json"],
        log=logs / "metadata.log", allow_fail=True)

    run([sys.executable, HERE / "generate_corpus.py", "--repo-root", repo,
         "--out-dir", corpus_dir, "--preset", args.preset, "--outer-steps", outer],
        log=logs / "generate_corpus.log")
    ref = corpus_dir / "reference.json"
    corpus = corpus_dir / "operators_i64.bin"
    K, modulus_bits = exact_k_for_reference(ref)
    if args.quick:
        batches = [1, 2]
    elif args.gpu_batches.strip().lower() == "auto":
        sat = max(1, min(32, math.ceil(8192 / K)))
        batches = sorted(set([1, sat]))
    else:
        batches = parse_batches(args.gpu_batches)
    capacity = {"K": K, "prime_bits": 30, "modulus_bits": modulus_bits,
                "criterion": "product(primes) > 2*max_abs(exact_reference)"}
    (work / "rns_capacity.json").write_text(json.dumps(capacity, indent=2) + "\n")
    print("RNS exact capacity:", json.dumps(capacity))

    build = work / "build"
    run(["cmake", "-S", HERE / "native", "-B", build, "-DCMAKE_BUILD_TYPE=Release"],
        log=logs / "cmake_configure.log")
    run(["cmake", "--build", build, "-j", str(args.threads)], log=logs / "cmake_build.log")

    py_backends = "numpy,flint"
    if not args.skip_python_slow:
        py_backends = "python_int,sympy,flint,gmpy2,numpy"
    run([sys.executable, HERE / "bench_python.py", corpus,
         "--backends", py_backends, "--repeats", str(repeats), "--warmups", str(warmups),
         "--flint-threads", "1", "--output", work / "python_results.jsonl"],
        log=logs / "bench_python.log", allow_fail=True)

    native = build / "bench_native"
    exact_text_dumps = []
    native_rows = []
    native_jobs = [
        ("boost", 1),
        ("gmp", 1),
        ("flint", 1),
        ("flint", args.threads),
    ]
    for backend, th in native_jobs:
        tag = f"{backend}_t{th}"
        dump = dumps / f"{tag}.txt"
        p = run([native, corpus, "--backend", backend, "--threads", str(th),
                 "--repeats", str(repeats), "--warmups", str(warmups), "--dump", dump],
                log=logs / f"{tag}.log", allow_fail=True)
        if p.returncode == 0:
            exact_text_dumps.append(dump)
            native_rows.extend(x for x in p.stdout.splitlines() if x.startswith("{"))

    for th in sorted(set([1, args.threads])):
        tag = f"rns_cpu_t{th}"
        dump = dumps / f"{tag}.bin"
        p = run([native, corpus, "--backend", "rns_cpu", "--threads", str(th), "--k", str(K),
                 "--repeats", str(repeats), "--warmups", str(warmups), "--dump", dump],
                log=logs / f"{tag}.log", allow_fail=True)
        if p.returncode == 0:
            native_rows.extend(x for x in p.stdout.splitlines() if x.startswith("{"))

    accel_dump = dumps / "accelerate.bin"
    p = run([native, corpus, "--backend", "accelerate", "--repeats", str(repeats),
             "--warmups", str(warmups), "--dump", accel_dump],
            log=logs / "accelerate.log", allow_fail=True)
    if p.returncode == 0:
        native_rows.extend(x for x in p.stdout.splitlines() if x.startswith("{"))
    (work / "native_results.jsonl").write_text("\n".join(native_rows) + ("\n" if native_rows else ""))

    metal = build / "bench_metal"
    metal_rows = []
    rns_dumps, f64_dumps = [], []
    if metal.exists():
        for backend in ["rns_gpu", "df64_gpu"]:
            for batch in batches:
                tag = f"{backend}_b{batch}"
                dump = dumps / (tag + (".rns" if backend == "rns_gpu" else ".f64"))
                argv = [metal, corpus, "--backend", backend, "--batch", str(batch),
                        "--repeats", str(repeats), "--warmups", str(warmups), "--dump", dump]
                if backend == "rns_gpu":
                    argv += ["--k", str(K)]
                p = run(argv, log=logs / f"{tag}.log", allow_fail=True)
                if p.returncode == 0:
                    metal_rows.extend(x for x in p.stdout.splitlines() if x.startswith("{"))
                    (rns_dumps if backend == "rns_gpu" else f64_dumps).append(dump)
    (work / "metal_results.jsonl").write_text("\n".join(metal_rows) + ("\n" if metal_rows else ""))

    # Verify every exact dump and quantify floating-point projective error.
    verify_argv = [sys.executable, HERE / "verify_results.py", "--reference", ref,
                   "--output", work / "verification.jsonl"]
    for x in exact_text_dumps: verify_argv += ["--int-text", x]
    for x in sorted(dumps.glob("rns_cpu_t*.bin")): verify_argv += ["--rns", x]
    for x in rns_dumps: verify_argv += ["--rns", x]
    if accel_dump.exists(): verify_argv += ["--f64", accel_dump]
    for x in f64_dumps: verify_argv += ["--f64", x]
    run(verify_argv, log=logs / "verification.log")

    if not args.skip_production:
        run([sys.executable, HERE / "benchmark_production_df64.py", "--repo-root", repo,
             "--count", str(2000 if args.quick else args.production_count),
             "--repeats", str(1 if args.quick else repeats),
             "--output", work / "production_df64.jsonl"],
            log=logs / "production_df64.log", allow_fail=True)

    manifest = {
        "label": args.label,
        "created_unix": time.time(),
        "preset": args.preset,
        "outer_steps": outer,
        "repeats": repeats,
        "warmups": warmups,
        "threads": args.threads,
        "gpu_batches": batches,
        "rns_capacity": capacity,
        "scientific_note": "Batch>1 Metal results replay the same exact operator chain to measure saturated kernel throughput; do not interpret them as unique-trajectory application throughput. production_df64.jsonl uses unique trajectories.",
    }
    (work / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nComplete: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
