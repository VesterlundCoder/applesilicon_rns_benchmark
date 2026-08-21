---
title: 'A reproducible exact/approximate benchmark for 6F5 conservative-matrix-field matrix chains on Apple Silicon'
tags:
  - Apple Silicon
  - Metal
  - residue number system
  - exact linear algebra
  - conservative matrix fields
  - Ramanujan Machine
  - benchmarking
authors:
  - name: David Vesterlund
    orcid: 0009-0000-6455-1141
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 21 August 2026
bibliography: paper.bib
---

# Summary

`apple6f5_benchmark` is a small, fully reproducible benchmark suite that measures
how fast a long chain of integer $6\times6$ matrix products can be evaluated on an
Apple-Silicon laptop, and how much precision is lost by the fast (floating-point)
methods that make the computation tractable at scale. The matrix chain is not
synthetic: every operator is a *primitive projective integer* matrix extracted
from the corrected Ramanujan Dreams $_6F_5$ conservative-matrix-field (CMF) walk
semantics [@elimelech2024constants; @weinbaum2025cmf; @raayoni2021ramanujan], so
the benchmark exercises the exact kernel that underlies an active line of
experimental-mathematics research. The suite ships three workloads
(`sparse_x1`, `x6`, `dense_pos`) spanning 17- to 62-bit operators, and is
archived at DOI 10.5281/zenodo.22045418 and
`https://github.com/VesterlundCoder/applesilicon_rns_benchmark`
[@vesterlund2026artifact].

The suite generates a deterministic, SHA-256-pinned corpus of operators together
with an exact big-integer reference product, then times a broad set of back-ends
against that reference on the same input:

- **Exact CPU:** Python `int`, SymPy [@sympy], python-flint / FLINT
  `fmpz_mat_mul` [@flint], GMP `mpz_class` [@gmp], Boost.Multiprecision
  `cpp_int` [@boost], and a threaded residue-number-system (RNS) engine using
  32-bit Montgomery multiplication [@montgomery1985] with Chinese-Remainder
  reconstruction [@knuth1997].
- **Exact GPU:** a Metal compute kernel that runs the same RNS chain, one
  `(prime, trajectory)` lane per thread.
- **Approximate:** Apple Accelerate `cblas_dgemm` [@accelerate], NumPy [@numpy],
  and a Metal *double-float* (df64) kernel [@dekker1971; @hida2001qd] with
  power-of-two renormalisation after every operator, reported as projective
  error against the exact reference.

Every exact back-end must reproduce the reference SHA-256 or the run fails;
every floating-point back-end reports its worst-case projective absolute and
relative error. A separate driver benchmarks the repository's production Metal
df64 CMF walk on *unique* deterministic trajectories, so that saturated-kernel
throughput is never confused with real application throughput.

# Statement of need

The Ramanujan Machine programme and its conservative-matrix-field
generalisation search enormous spaces of matrix-valued recurrences for new
closed forms of mathematical constants [@raayoni2021ramanujan; @elimelech2024constants].
The inner loop is always the same: multiply a long, structured chain of small
rational/integer matrices and inspect the projective limit. Two facts make this
loop a genuine engineering problem rather than a triviality:

1. The accumulated product grows to thousands of bits after only a few hundred
   steps (9546 bits after 1000 unit-axis $_6F_5$ operators in our default
   workload), so naive exact arithmetic is expensive and floating point silently
   loses the very digits the search depends on.
2. Modern experimental-mathematics pipelines run these walks *by the million* on
   consumer Apple-Silicon hardware, where the fastest path (Accelerate BLAS, or a
   Metal df64 kernel) is only *approximately* correct, and the exact path (GMP,
   FLINT, or an RNS decomposition) trades single-chain latency for GPU throughput.

Despite this, there is no shared, independently checkable benchmark that answers
the practical questions a researcher actually asks: *How much faster is the
approximate path? How much precision does it cost? When does the GPU win? Is a
residue-number-system decomposition competitive with FLINT for this matrix
size?* `apple6f5_benchmark` provides exactly that — a single `run_all.py` that
builds the native/Metal binaries, generates the pinned corpus, runs all
back-ends with warm-ups and repeat medians, verifies correctness, and writes
machine-readable JSON plus exact dump files that a third party can re-check
without trusting our timings.

The suite is deliberately self-contained and hardware-honest: it records the
full system profile (chip, core split, compiler, BLAS provenance, git commit),
distinguishes GPU-native from host-side timing, and separates three throughput
measures that are easy to conflate: RNS **modular-kernel** throughput (residues
only), **end-to-end exact** throughput (including parallel Chinese-Remainder
reconstruction), and **unique-trajectory** application throughput. A CPU
trajectory-parallel mode (GMP/FLINT/Accelerate across all cores) makes the
CPU/GPU throughput comparison fair rather than pitting a saturated GPU against a
single CPU thread. It is intended both as a reproducibility artefact for papers built
on the Ramanujan Dreams engine and as a template others can adapt to benchmark
exact-vs-approximate small-matrix chains on their own machines.

# Functionality

Running

```
python run_all.py --repo-root <RamanujanDreams-RNS-Mac> --label M5-Max
```

produces, for the default `sparse_x1` workload (1000 operators, $z=1/2$, rank 6):

- `corpus/` — the SHA-256-pinned operator binary, exact reference, and metadata;
- `python_results.jsonl`, `native_results.jsonl`, `metal_results.jsonl` — timing
  medians (plus MAD, min, max) for every back-end;
- `verification.jsonl` — per-dump correctness (exact) or projective error
  (floating), including the RNS modulus-capacity certificate;
- `production_df64.jsonl` — unique-trajectory application throughput;
- `system_metadata.json`, `run_manifest.json` — full provenance.

On an Apple M5 Max (18-core CPU: 6 super + 12 performance cores; 40-core GPU
[@applem5max]) the exact single-chain result is fastest with GMP ($\sim$8.5 ms),
the approximate single-chain result is fastest with Accelerate ($\sim$0.27 ms at
$2.4\times10^{-15}$ projective error), and the RNS-GPU modular kernel saturates
at $\sim$3k residue-trajectories/GPU-second. Crucially, once per-trajectory
Chinese-Remainder reconstruction is included the exact RNS path
($\sim$0.5k/s) falls below trajectory-parallel CPU GMP ($\sim$1.5k/s), so the
GPU wins on exact arithmetic only when reconstruction is amortised. All numbers
are regenerated end-to-end by the command above.

# Acknowledgements

We thank the FLINT, GMP, Boost, and NumPy communities, whose libraries are the
exact-arithmetic baselines in this work.

# References
