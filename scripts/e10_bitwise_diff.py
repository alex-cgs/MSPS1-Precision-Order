"""E10: where the numpy and JAX GPU kernels differ. Same member ICs, fixed step, per-member comparison after 32, 320, 3200 steps.

usage (env with JAX): e10_bitwise_diff.py [BATCH]. Writes results/derived/e10_bitwise_diff.csv: members differing and the largest
absolute difference per case, precision, order and step count. Also whether a member differs when run alone, to separate an IC effect from a batch effect.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MSPS_ROOT", str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

from msps.gpu.kernels import BatchedTaylor  # noqa: E402
from msps.systems import load_case  # noqa: E402

jax.config.update("jax_enable_x64", True)
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 100
STEP_H = {"lorenz63": 1 / 64, "nb3_ensemble": 1 / 64}
CHECKS = (32, 320, 3200)


def run(case, n, prec, backend, states):
    k = BatchedTaylor(case.tape, n, prec, backend)
    z = k.from_fractions(states)
    h = k.xp.full((len(states),), STEP_H[case.name], dtype=k.xp.float64)
    out = {}
    for s in range(1, CHECKS[-1] + 1):
        z = k.step(z, h)
        if s in CHECKS:
            a = k.to_numpy(z)
            out[s] = (np.asarray(a[0], dtype=np.float64) + np.asarray(a[1], dtype=np.float64)) if isinstance(a, tuple) else np.asarray(a, dtype=np.float64)
            out[s, "raw"] = np.stack(a) if isinstance(a, tuple) else a[None]
    return out


rows = []
for spec in ("lorenz63:rho_28", "nb3_ensemble:r4"):
    case = load_case(spec)
    states = [case.ic(i) for i in range(BATCH)]
    for n in (8, 16, 24):
        for prec in ("f32", "f64", "dd"):
            with np.errstate(all="ignore"):
                c, g = run(case, n, prec, "numpy", states), run(case, n, prec, "jax", states)
                for s in CHECKS:
                    # member axis is last; a member differs if any bit of any state component differs (NaN compares equal to NaN here)
                    same = np.array([np.array_equal(c[s, "raw"][..., m], g[s, "raw"][..., m], equal_nan=True) for m in range(BATCH)])
                    d = np.abs(c[s] - g[s]).max(axis=0)
                    fin = np.isfinite(d)
                    rows.append({"case": spec, "n": n, "prec": prec, "steps": s, "members": BATCH, "members_differing": int((~same).sum()), "first_differing_member": int(np.argmin(same)) if (~same).any() else -1,
                                 "members_non_finite": int((~fin).sum()), "max_abs_diff_finite": float(d[fin].max()) if fin.any() else float("nan"), "median_abs_diff_finite": float(np.median(d[fin])) if fin.any() else float("nan")})
                    print(rows[-1], flush=True)
                # same members one at a time, first 10, 32 steps
                alone = []
                for m in range(min(10, BATCH)):
                    k = BatchedTaylor(case.tape, n, prec, "numpy")
                    kg = BatchedTaylor(case.tape, n, prec, "jax")
                    za, zb = k.from_fractions([states[m]]), kg.from_fractions([states[m]])
                    ha, hb = k.xp.full((1,), STEP_H[case.name], dtype=k.xp.float64), kg.xp.full((1,), STEP_H[case.name], dtype=kg.xp.float64)
                    for _ in range(32):
                        za, zb = k.step(za, ha), kg.step(zb, hb)
                    a, b = k.to_numpy(za), kg.to_numpy(zb)
                    a, b = (np.stack(a), np.stack(b)) if isinstance(a, tuple) else (a, b)
                    alone.append(int(not np.array_equal(a, b, equal_nan=True)))
                print({"case": spec, "n": n, "prec": prec, "alone_first10_differ_after_32": alone}, flush=True)
                for r in rows:
                    if r["case"] == spec and r["n"] == n and r["prec"] == prec:
                        r["alone_first10_differing"] = sum(alone)
out = ROOT / "results" / "derived" / "e10_bitwise_diff.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("written", out)

# Elementary f32 and f64 operations on the GPU against numpy (IEEE, correctly rounded), 2e6 random operands.
rng = np.random.default_rng(20261107)
ops = [("add", lambda a, b: a + b), ("mul", lambda a, b: a * b), ("div", lambda a, b: a / b), ("div_by_int", None)]
rows2 = []
for name, dt, jt in (("f32", np.float32, jax.numpy.float32), ("f64", np.float64, jax.numpy.float64)):
    a = (rng.uniform(1, 2, 2_000_000) * 2.0 ** rng.integers(-20, 20, 2_000_000)).astype(dt)
    b = (rng.uniform(1, 2, 2_000_000) * 2.0 ** rng.integers(-20, 20, 2_000_000)).astype(dt)
    k = rng.integers(1, 41, 2_000_000).astype(dt)
    for op, fn in ops:
        y = k if fn is None else b
        f = fn or (lambda p, q: p / q)
        cpu = f(a, y)
        gpu = np.asarray(jax.jit(f)(jax.numpy.asarray(a), jax.numpy.asarray(y)))
        rows2.append({"prec": name, "op": op, "operands": a.size, "differing": int((cpu != gpu).sum()), "max_ulp": int(np.abs(cpu.view("i4" if name == "f32" else "i8").astype(np.int64) - gpu.view("i4" if name == "f32" else "i8").astype(np.int64)).max())})
        print(rows2[-1], flush=True)
with open(ROOT / "results" / "derived" / "e10_gpu_elementary_ops.csv", "w", newline="") as f_:
    w = csv.DictWriter(f_, fieldnames=list(rows2[0]))
    w.writeheader()
    w.writerows(rows2)
