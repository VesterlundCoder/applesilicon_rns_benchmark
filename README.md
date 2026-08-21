# Apple Silicon 6F5 CMF Benchmark Suite

Reproducible benchmark code for the corrected Ramanujan Dreams 6F5 `Matrix.walk`
workload on Apple Silicon CPU and Metal GPU.

The suite is designed for two separate scientific questions:

1. **Cross-backend 6x6 matrix-chain benchmark.** Every backend receives the same
   deterministic sequence of 6x6 operators derived from the corrected 6F5 walk.
2. **Application benchmark.** The repository's existing corrected df64 Metal walk is
   measured on unique canonical trajectories at `z=1/2`, rank 6, normally `N=1000`.

Do not merge these two result classes into one performance number. The production
walk exploits the special companion/axis structure and performs an O(r^2) right
update; the generic matrix-chain benchmark deliberately measures a common dense
6x6 multiplication task so FLINT/GMP/SymPy/RNS/BLAS can be compared fairly.

## Scientific workload

The source of truth is:

```text
RamanujanDreams-RNS-Mac/lib/cmf_walk_corrected.py
```

For every rational axis operator `A` in the corrected walk, `generate_corpus.py`
creates a primitive integer matrix

```text
A* = s A
```

where `s` is a nonzero rational scalar chosen by clearing denominators and removing
the common integer gcd. Since CMF limits are projective, the accumulated product
of the `A*` matrices differs from the true walk only by a nonzero scalar. The
projective matrix ratios are unchanged.

The publication workload should use:

```text
z = 1/2
rank = 6
shift = (1,...,1)
preset = sparse_x1
axis direction = (1,0,...,0)
outer steps = 1000
```

This gives exactly one genuine corrected 6F5 unit-axis operator per outer step and
keeps the cross-backend interpretation especially clean. Additional `x6` and
`dense_pos` workloads are included as secondary stress tests.

## Backends

### Exact

- Python built-in arbitrary precision `int`
- SymPy `Matrix`
- python-flint `fmpz_mat`
- native FLINT `fmpz_mat_mul`
- GMP through native `mpz_class`
- gmpy2 through Python
- native C++ Boost.Multiprecision `cpp_int`, compiled `-O3`
- RNS CPU, 30-bit prime lanes, Montgomery32 modular multiplication
- RNS Metal GPU, same 30-bit/Montgomery representation

For RNS, the number of primes `K` is not hand-picked. `run_all.py` computes the
smallest deterministic prime set whose product is greater than twice the largest
absolute element of the exact reference product. This is the signed-CRT capacity
condition required for exact reconstruction.

RNS output is reconstructed by CRT on the CPU and compared element-by-element with
the Python exact reference. The benchmark reports both modular-chain time and CRT
host time. A modular result that does not pass CRT verification is rejected.

### Approximate references

- NumPy `float64` matrix multiplication, power-of-two normalization after every
  operator
- Apple Accelerate `cblas_dgemm`, power-of-two normalization after every operator
- Metal df64/double-single, using the same two-float arithmetic pattern as the
  Ramanujan Dreams Metal kernel and exact power-of-two normalization after every
  operator

These are **not exact arithmetic competitors**. They are throughput/accuracy
references. The output is compared projectively with the exact reference and the
maximum absolute and relative projective errors are stored.

### LinBox / FFLAS-FFPACK

Not in the default table. FFLAS-FFPACK is most naturally a finite-field GEMM
baseline, so a scientifically clean comparison is against an individual RNS prime
lane or as the modular engine inside a full multi-prime + CRT pipeline. Treating a
single finite-field multiplication as if it were an exact arbitrary-integer result
would be misleading. Add it as a secondary modular-kernel experiment if desired.

## Installation on Apple Silicon

Apple Accelerate and Metal are part of macOS/Xcode. Install the remaining native
libraries with your normal package manager. A typical Homebrew environment is:

```bash
brew install cmake boost gmp flint pkg-config
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy this directory into the public Mac repository, for example:

```text
RamanujanDreams-RNS-Mac/
  benchmarks/
    apple6f5/
      README.md
      run_all.py
      ...
```

The scripts also work from another directory when `--repo-root` is supplied.

## First: quick correctness run

From this benchmark directory:

```bash
python run_all.py \
  --repo-root /path/to/RamanujanDreams-RNS-Mac \
  --label M1 \
  --workdir ./results \
  --quick
```

Run the same test on the other machine:

```bash
python run_all.py \
  --repo-root /path/to/RamanujanDreams-RNS-Mac \
  --label M5 \
  --workdir ./results \
  --quick
```

Do not proceed to publication timings until `verification.jsonl` reports
`"correct": true` for every exact backend.

## Publication run

Use the same benchmark commit and the same Ramanujan Dreams repository commit on
both machines:

```bash
python run_all.py \
  --repo-root /path/to/RamanujanDreams-RNS-Mac \
  --label M1 \
  --workdir ./results \
  --preset sparse_x1 \
  --outer-steps 1000 \
  --repeats 7 \
  --warmups 2
```

and:

```bash
python run_all.py \
  --repo-root /path/to/RamanujanDreams-RNS-Mac \
  --label M5 \
  --workdir ./results \
  --preset sparse_x1 \
  --outer-steps 1000 \
  --repeats 7 \
  --warmups 2
```

`--gpu-batches auto` is the default. It reports batch 1 latency plus one replay
batch chosen to give roughly 8192 RNS prime lanes, capped at batch 32. Batch > 1
replays the same matrix chain. It measures **saturated kernel throughput**, not
throughput over unique mathematical trajectories.

The separate `production_df64.jsonl` benchmark uses unique deterministic canonical
trajectories and is the application-throughput measurement.

## Production df64 timing patch

The existing repository host surrounds `[cb commit]` / `[cb waitUntilCompleted]`
with a CPU wall clock. For a paper, use Metal command-buffer GPU timestamps so GPU
execution is not conflated with CPU wait/submission overhead.

A small patch is included:

```bash
git -C /path/to/RamanujanDreams-RNS-Mac apply \
  /path/to/apple6f5/production_timing_patch.diff
```

Then rebuild the main repository before the publication run.

The benchmark's own Metal executable already reports both:

```text
gpu_median_s   = GPUEndTime - GPUStartTime
wall_median_s  = CPU commit/wait wall time
```

## Results layout

For each machine label:

```text
results/<LABEL>/
  system_metadata.json
  run_manifest.json
  rns_capacity.json
  corpus/
    operators_i64.bin
    metadata.json
    reference.json
  python_results.jsonl
  native_results.jsonl
  metal_results.jsonl
  verification.jsonl
  production_df64.jsonl
  dumps/
  logs/
```

`system_metadata.json` records the Apple hardware report, macOS version, compiler,
Python/package versions, NumPy build/BLAS configuration, and the exact git commit
and worktree status of the Ramanujan Dreams checkout.

## Recommended paper tables

### Table A: exact matrix-chain latency

Report for each M1/M5 system:

```text
Backend | exact? | threads | time | CRT time | exact total | correctness
```

For RNS, use `exact_result_median_s` on CPU and
`exact_result_wall_plus_crt_median_s` on GPU when comparing time-to-exact-result.
Also report the raw GPU kernel time separately.

### Table B: approximate reference

```text
Backend | time | normalization | max projective abs error | max projective rel error
```

Never label NumPy/Accelerate/df64 as exact.

### Table C: real Ramanujan Dreams application throughput

```text
Chip | GPU | N | z | trajectories | OK fraction | GPU traj/s | wall traj/s
```

Use the corrected production walk only.

### Figure: exact-cost crossover

Repeat the corpus benchmark for prefixes such as:

```text
N = 10, 25, 50, 100, 250, 500, 1000
```

Plot time against exact reference bit width, not only `N`. This shows where RNS
becomes competitive as integer growth increases.

## Experimental protocol for M1 vs M5

For defensible numbers:

1. Use AC power and disable Low Power Mode.
2. Close unrelated compute-heavy applications.
3. Reboot or otherwise start from a controlled idle state.
4. Record ambient conditions if thermal throttling becomes visible.
5. Run correctness first.
6. Use at least two warmups and seven timed repetitions for the final table.
7. Keep corpus SHA-256, git commit, compiler flags, package versions and macOS
   version in the archived result bundle.
8. Run the machines separately with the same corpus and software commit.
9. Report median and MAD, not only the fastest run.
10. Do not compare RNS modular-kernel time directly with FLINT time-to-exact-result
    without also showing CRT.

For stronger statistics, perform several independent benchmark sessions on each
machine (for example on three separate cold-start sessions) and model session as a
blocking factor rather than treating repeated measurements in one process as fully
independent observations.

## Files

- `generate_corpus.py`: derive the common 6F5 operator chain from the exact oracle
- `bench_python.py`: Python int, SymPy, python-flint, gmpy2, NumPy
- `native/bench_native.cpp`: Boost, GMP, FLINT, Accelerate, RNS CPU
- `native/bench_metal.mm`: Metal host and GPU timestamping
- `native/bench_kernels.metal`: RNS Montgomery32 and dense df64 Metal kernels
- `verify_results.py`: CRT reconstruction and exact/projective verification
- `benchmark_production_df64.py`: real corrected 6F5 df64 application benchmark
- `collect_metadata.py`: reproducibility metadata
- `run_all.py`: end-to-end orchestrator
- `production_timing_patch.diff`: correct GPU-only timing for the existing host
