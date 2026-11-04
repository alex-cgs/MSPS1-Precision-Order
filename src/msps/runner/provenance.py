"""Provenance on every run (spec §9, rule 2). Cached per process; git and hardware do not change within a job."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time

from ..systems import ROOT

_cache: dict | None = None


def _sh(*cmd, cwd=None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=20).stdout.strip()
    except Exception:
        return ""


def provenance(seed=None) -> dict:
    global _cache
    if _cache is None:
        import gmpy2
        import numpy
        import scipy
        import sympy
        try:
            import flint
            flint_v = flint.__version__
        except ImportError:
            flint_v = None
        try:
            import jax
            jax_v = jax.__version__
        except Exception:
            jax_v = None
        cpu = ""
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
        gpu = _sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
        _cache = {
            "git_commit": _sh("git", "rev-parse", "HEAD", cwd=ROOT),
            "git_dirty": bool(_sh("git", "status", "--porcelain", "--", "src", "cases", "experiments", cwd=ROOT)),
            "python": sys.version.split()[0],
            "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__,
                         "gmpy2": gmpy2.version(), "mpfr": gmpy2.mpfr_version(),
                         "python-flint": flint_v, "jax": jax_v},
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": cpu,
            "fma": " fma " in flags,
            "gpu": gpu,
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        }
    out = dict(_cache)
    out["seed"] = seed
    out["pid"] = os.getpid()
    return out


def stamp() -> float:
    return time.time()
