#!/usr/bin/env python3
"""Generate the two Frontiers figures from the sweep outputs.

Figure 1  batch-scaling throughput (log-log): exact vs approximate, CPU vs GPU.
Figure 2  bit-growth / crossover: bits(P_N) and K(N) with backend times vs N.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def fig_batch(path: Path, out: Path, label: str):
    rows = load_jsonl(path)
    families = {
        "rns_gpu_kernel": ("RNS-GPU modular kernel (exact-modular)", "o", "-"),
        "rns_exact_end_to_end_crt": ("RNS end-to-end exact (kernel+CRT)", "s", "--"),
        "cpu_gmp": ("CPU GMP exact (trajectory-parallel)", "^", "-"),
        "df64_gpu": ("df64-GPU (approx)", "D", "-"),
        "cpu_accel": ("CPU Accelerate (approx, trajectory-parallel)", "v", "-"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for fam, (lbl, mk, ls) in families.items():
        pts = sorted(((r["batch"], r["throughput_per_s"]) for r in rows
                      if r["family"] == fam and r.get("throughput_per_s")), key=lambda x: x[0])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=mk, linestyle=ls, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B (independent trajectories)")
    ax.set_ylabel("throughput (trajectories / s)")
    ax.set_title(f"6F5 chain throughput vs batch size ({label}, N=1000)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


def fig_bitgrowth(path: Path, out: Path, label: str):
    rows = sorted(load_jsonl(path), key=lambda r: r["N"])
    N = [r["N"] for r in rows]
    bits = [r["bits"] for r in rows]
    K = [r["K"] for r in rows]

    fig, axL = plt.subplots(figsize=(7.2, 5.0))
    l1 = axL.plot(N, bits, marker="o", color="C0", label="bits(P_N) (largest entry)")
    axL.set_xlabel("chain length N (operators)")
    axL.set_ylabel("bits / RNS primes K", color="C0")
    l2 = axL.plot(N, K, marker="s", color="C1", label="K(N) RNS primes")
    axL.tick_params(axis="y", labelcolor="C0")
    axL.grid(True, alpha=0.3)

    axR = axL.twinx()
    lines = list(l1) + list(l2)
    for key, lbl, mk, col in (
        ("t_gmp_s", "GMP time", "^", "C2"),
        ("t_flint_s", "FLINT time", "v", "C3"),
        ("t_rns_exact_result_s", "RNS end-to-end exact (kernel+CRT)", "D", "C4"),
    ):
        pts = [(r["N"], r[key]) for r in rows if r.get(key)]
        if pts:
            xs, ys = zip(*pts)
            lines += axR.plot(xs, ys, marker=mk, linestyle="--", color=col, label=lbl)
    axR.set_yscale("log")
    axR.set_ylabel("single-result time (s)")
    axL.set_title(f"Bit growth and exact-arithmetic crossover ({label})")
    axR.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-jsonl", type=Path, required=True)
    ap.add_argument("--bitgrowth-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", default="M5-Max")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_batch(args.batch_jsonl, args.out_dir / "fig_batch_scaling", args.label)
    fig_bitgrowth(args.bitgrowth_jsonl, args.out_dir / "fig_bitgrowth_crossover", args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
