#!/usr/bin/env python3
"""Collect JSONL benchmark outputs into one CSV for analysis/paper plotting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+", type=Path,
                    help="e.g. results/M1 results/M5")
    ap.add_argument("--output", type=Path, default=Path("benchmark_summary.csv"))
    args = ap.parse_args()
    allrows = []
    for d in args.result_dirs:
        label = d.name
        for fn, family in [
            ("python_results.jsonl", "python"),
            ("native_results.jsonl", "native"),
            ("metal_results.jsonl", "metal"),
            ("verification.jsonl", "verification"),
            ("production_df64.jsonl", "production"),
        ]:
            for row in read_jsonl(d / fn):
                row = dict(row)
                row["machine_label"] = label
                row["result_family"] = family
                row.pop("stderr", None)
                allrows.append(row)
    fields = sorted({k for r in allrows for k in r})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(allrows)
    print(f"wrote {len(allrows)} rows to {args.output}")


if __name__ == "__main__":
    main()
