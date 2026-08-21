#!/usr/bin/env python3
"""Collect machine, OS, compiler, BLAS, package and repository metadata."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def cmd(argv):
    try:
        p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return {"command": argv, "returncode": p.returncode, "output": p.stdout.strip()}
    except Exception as exc:
        return {"command": argv, "error": f"{type(exc).__name__}: {exc}"}


def pkg(name):
    try:
        mod = __import__(name)
        if name == "gmpy2":
            return mod.version()
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    obj = {
        "label": args.label,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "packages": {x: pkg(x) for x in ["numpy", "sympy", "flint", "gmpy2"]},
        "commands": {},
    }
    commands = {
        "sw_vers": ["sw_vers"],
        "uname": ["uname", "-a"],
        "hardware": ["system_profiler", "SPHardwareDataType", "-json"],
        "display": ["system_profiler", "SPDisplaysDataType", "-json"],
        "clang": ["clang++", "--version"],
        "cmake": ["cmake", "--version"],
        "sysctl_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "sysctl_mem": ["sysctl", "-n", "hw.memsize"],
        "sysctl_perfcores": ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
        "sysctl_effcores": ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
    }
    for key, argv in commands.items():
        obj["commands"][key] = cmd(argv)
    obj["numpy_config"] = cmd([sys.executable, "-c", "import numpy as np; np.show_config()"])
    if args.repo_root:
        root = str(args.repo_root.resolve())
        obj["git"] = {
            "head": cmd(["git", "-C", root, "rev-parse", "HEAD"]),
            "status": cmd(["git", "-C", root, "status", "--porcelain=v1"]),
            "remote": cmd(["git", "-C", root, "remote", "-v"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
