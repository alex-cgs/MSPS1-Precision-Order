"""E10 hardware benchmark (spec §11). Throughput in trajectory-steps/s with warm-up excluded, >= 5 repeats, accuracy
reached (T_pred on the case's reference, when cached), bitwise fingerprints for cross-device comparison.

usage: e10_bench.py --label ops_h100 --size small [--backend jax|numpy] [--deadline-guard] [--systems ...]
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MSPS_ROOT", str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from msps.gpu.kernels import BatchedTaylor  # noqa: E402
from msps.io import write_run  # noqa: E402
from msps.runner.provenance import provenance  # noqa: E402
from msps.runner.window import Window  # noqa: E402
from msps.systems import load_case  # noqa: E402

GRIDS = {
    "tiny": {"batch": [10, 100], "n": [8], "prec": ["f32", "f64", "dd"], "repeats": 3},
    "small": {"batch": [100, 1000, 10000], "n": [8, 16], "prec": ["f32", "f64", "dd"], "repeats": 5},
    "full": {"batch": [1, 10, 100, 1000, 10000, 100000], "n": [8, 16, 24], "prec": ["f32", "f64", "dd"], "repeats": 5},
}
STEP_H = {"lorenz63": 1 / 64, "nb3_ensemble": 1 / 64}


def ensemble_states(case, batch):
    """Available ICs repeated with a tiny rational jitter so members stay distinct at large batch."""
    base = [case.ic(i) for i in range(min(batch, 100))]
    out = []
    for m in range(batch):
        s = list(base[m % len(base)])
        if m >= len(base):
            s[0] = s[0] + Fraction(m, 2**40)
        out.append(s)
    return out


def fingerprint(arr):
    if isinstance(arr, tuple):
        return hashlib.sha1(np.asarray(arr[0]).tobytes() + np.asarray(arr[1]).tobytes()).hexdigest()
    return hashlib.sha1(np.asarray(arr).tobytes()).hexdigest()


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def peak_mem():
    try:
        import jax
        s = jax.devices()[0].memory_stats() or {}
        return s.get("peak_bytes_in_use")
    except Exception:
        return None


def bench_config(case, n, prec, batch, backend, repeats, steps, label, window, tracer, device_tag):
    kern = BatchedTaylor(case.tape, n, prec, backend)
    states = ensemble_states(case, batch)
    z0 = kern.from_fractions(states)
    xp = kern.xp
    h_val = STEP_H[case.name]
    h = xp.full((batch,), h_val, dtype=xp.float64)
    t0 = time.perf_counter()
    z = kern.step(z0, h)
    (z[0] if isinstance(z, tuple) else z).block_until_ready() if hasattr((z[0] if isinstance(z, tuple) else z), "block_until_ready") else None
    compile_s = time.perf_counter() - t0
    walls = []
    for r in range(repeats):
        if window is not None and window.must_stop(60):
            break
        t0 = time.perf_counter()
        z = z0
        for _ in range(steps):
            z = kern.step(z, h)
        hd = z[0] if isinstance(z, tuple) else z
        if hasattr(hd, "block_until_ready"):
            hd.block_until_ready()
        walls.append(time.perf_counter() - t0)
    arr = kern.to_numpy(z)
    throughput = [batch * steps / w for w in walls]
    return {
        "case": case.id, "n": n, "prec": prec, "batch": batch, "steps": steps, "repeats": len(walls),
        "walls": walls, "wall_median": float(np.median(walls)) if walls else None, "compile_s": compile_s,
        "throughput_median": float(np.median(throughput)) if throughput else None,
        "throughput_iqr": [float(np.percentile(throughput, 25)), float(np.percentile(throughput, 75))] if throughput else None,
        "peak_bytes": peak_mem(), "device": device_tag, "fingerprint": fingerprint(arr),
    }


def accuracy(case, n, prec, backend, n_ic, device_tag):
    """T_pred (tol 1e-6, 1e-3) per member on the grid, comparing against cached E2 references; adaptive per-member."""
    from msps.runner.references import Reference, ref_dir
    def usable(i):  # a stored reference that failed its check raises on load (nb3_ensemble:r4 #65)
        p = ref_dir(case, i) / "meta.json"
        return p.exists() and json.loads(p.read_text()).get("check_passed")
    ics = [i for i in range(n_ic) if usable(i)]
    if not ics:
        return None
    refs = [Reference(case, i) for i in ics]
    grid_all = refs[0].grid_times
    grid = grid_all[3::4]  # every fourth grid point keeps the host transfers cheap
    ref_arr = np.array([[[float(g[j]) for j in case.system.positions] for g in r.grid()[3::4]] for r in refs])
    kern = BatchedTaylor(case.tape, n, prec, backend)
    states = [case.ic(i) for i in ics]
    z = kern.from_fractions(states)
    t = kern.xp.zeros(len(ics), dtype=kern.xp.float64)
    tols = [1e-6, 1e-3]
    hit = {tol: np.full(len(ics), np.nan) for tol in tols}
    idx = list(case.system.positions)
    t0 = time.perf_counter()
    total_steps = np.zeros(len(ics))
    for g, tg in enumerate(grid):
        z, t, s = kern.advance(z, t, float(tg))
        total_steps += np.asarray(s)
        zn = kern.to_numpy(z)
        q = (zn[0][idx].astype(np.longdouble) + zn[1][idx]) if isinstance(zn, tuple) else zn[idx].astype(np.longdouble)
        E = np.abs(q.T - ref_arr[:, g].astype(np.longdouble)).max(axis=1)
        E = np.where(np.isfinite(E), E, np.inf).astype(np.float64)
        for tol in tols:
            newly = np.isnan(hit[tol]) & (E > tol)
            hit[tol][newly] = float(tg)
        if np.all(~np.isnan(hit[1e-3])):
            break
    wall = time.perf_counter() - t0
    T = float(grid[-1])
    out = {"case": case.id, "n": n, "prec": prec, "device": device_tag, "members": len(ics), "wall": wall, "steps_total": float(total_steps.sum())}
    for tol in tols:
        cens = np.isnan(hit[tol])
        vals = np.where(cens, T, hit[tol])
        out[f"tpred_median_{tol:g}"] = float(np.median(vals))
        out[f"censored_{tol:g}"] = float(cens.mean())
    out["cost_per_accurate_time_s"] = wall / max(len(ics) * out["tpred_median_1e-06"], 1e-30)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--size", default="small", choices=list(GRIDS))
    ap.add_argument("--backend", default="jax", choices=["jax", "numpy"])
    ap.add_argument("--systems", nargs="*", default=["lorenz63:rho_28", "nb3_ensemble:r4"])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--deadline-guard", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--precs", nargs="*")
    a = ap.parse_args()
    window = Window() if a.deadline_guard else None
    if window is not None and not window.may_start(120):
        print("window guard: refusing to start", flush=True)
        return 3
    g = dict(GRIDS[a.size])
    if a.precs:
        g["prec"] = a.precs
    device_tag = a.label
    if a.backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
        device_tag += ":" + str(jax.devices()[0])
    out_rows, acc_rows = [], []
    env = {"gpu": gpu_info(), "host": socket.gethostname(), "provenance": provenance(), "size": a.size, "backend": a.backend}
    study = "E10"
    run_key = hashlib.sha1(f"{a.label}|{a.size}|{a.backend}|{a.systems}|{a.precs}".encode()).hexdigest()[:16]
    t_start = time.time()

    def flush(status):
        meta = {"label": a.label, "status": status, "env": env, "rows": out_rows, "accuracy": acc_rows, "t_start": t_start, "t_end": time.time(),
                "config": vars(a), "study": study, "run_id": run_key}
        write_run(study, run_key, meta)

    status = "completed"
    for spec in a.systems:
        case = load_case(spec)
        for n in g["n"]:
            for prec in g["prec"]:
                for batch in g["batch"]:
                    if window is not None and not window.may_start(120):
                        status = "stopped_by_window"
                        flush(status)
                        print("stopped by window", flush=True)
                        return 0
                    try:
                        row = bench_config(case, n, prec, batch, a.backend, g["repeats"], a.steps, a.label, window, None, device_tag)
                    except Exception as e:  # out of memory on the smaller card is a result, not a crash
                        row = {"case": case.id, "n": n, "prec": prec, "batch": batch, "error": repr(e)[:300], "device": device_tag}
                    out_rows.append(row)
                    flush("running")
                    print(json.dumps({k: row.get(k) for k in ("case", "n", "prec", "batch", "throughput_median", "compile_s", "error")}), flush=True)
            if not a.no_accuracy:
                for prec in g["prec"]:
                    if window is not None and not window.may_start(300):
                        status = "stopped_by_window"
                        flush(status)
                        return 0
                    try:
                        acc = accuracy(case, n, prec, a.backend, 100, device_tag)
                    except Exception as e:
                        acc = {"case": case.id, "n": n, "prec": prec, "error": repr(e)[:300]}
                    if acc:
                        acc_rows.append(acc)
                        flush("running")
                        print(json.dumps(acc), flush=True)
    flush(status)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
