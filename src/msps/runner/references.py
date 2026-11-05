"""Validated reference runs, cached by (case, ic, T, protocol hash) (spec §4.7, §9 rule 4)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import time
from fractions import Fraction
from pathlib import Path

import gmpy2

from ..arith import get_arith
from ..diagnostics import error_positions
from ..integrators.taylor import TaylorStepper
from ..io import RESULTS, _atomic_json
from ..schedules import AdaptiveDyadic
from ..systems import load_case
from .provenance import provenance

_loaded: dict = {}


def grid_dt(T: Fraction, override=None) -> Fraction:
    """Coarse fixed grid spacing: dyadic, 256–512 points on [0, T], unless the card overrides it."""
    if override is not None:
        return Fraction(str(override))
    return Fraction(2) ** math.floor(math.log2(float(T) / 256))


def protocol(case) -> dict:
    return {**case.card["reference"], "T": str(case.card["T"]), "grid_dt": str(grid_dt(Fraction(str(case.card["T"])), case.card["reference"].get("grid_dt")))}


def ref_dir(case, ic_id: int) -> Path:
    h = hashlib.sha1(json.dumps(protocol(case), sort_keys=True).encode()).hexdigest()[:12]
    return RESULTS / "references" / case.id.replace(":", "__") / str(ic_id) / h


def _dump(z):
    return [gmpy2.to_binary(x) for x in z]


def _load(z):
    return [gmpy2.from_binary(x) for x in z]


def compute_reference(case_spec: str, ic_id: int, force: bool = False) -> dict:
    case = load_case(case_spec)
    d = ref_dir(case, ic_id)
    if (d / "meta.json").exists() and not force:
        return json.loads((d / "meta.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    p = case.card["reference"]
    T = Fraction(str(case.card["T"]))
    dt = grid_dt(T, p.get("grid_dt"))
    z0 = case.ic(ic_id)
    t0 = time.time()
    arith = get_arith(f"mpfr-{p['bits']}")
    arith.activate()
    st = TaylorStepper(case, p["order"], arith)
    sched = AdaptiveDyadic(float(p["safety"]), Fraction(str(p["cap"])))
    z = [arith.convert(v) for v in z0]
    steps, states, rhos = [], [_dump(z)], []
    grid_times = [dt * k for k in range(1, int(T / dt) + 1)]
    grid_states = {}
    gi, t = 0, Fraction(0)
    dmin = float("inf")
    pairs = case.system.meta.get("pairs")
    max_steps = p.get("max_steps")
    while t < T:
        if max_steps is not None and len(steps) >= int(max_steps):
            raise RuntimeError(f"reference {case.id} #{ic_id} exceeded max_steps={max_steps} at t={float(t):.6g}")
        c = st.coeffs(z)
        rho = st.radius(c)
        h = min(sched.next_h(t, z, c, st), T - t)
        hc = arith.convert(h)
        while gi < len(grid_times) and grid_times[gi] <= t + h:
            grid_states[gi] = _dump(st.horner(c, arith.convert(grid_times[gi] - t)))
            gi += 1
        z = st.horner(c, hc)
        t += h
        steps.append(h)
        rhos.append(rho)
        states.append(_dump(z))
        if pairs:
            N, dim = case.system.meta["N"], case.system.meta["dim"]
            zf = [float(x) for x in z[:dim * N]]
            dmin = min(dmin, min(math.sqrt(sum((zf[a * N + i] - zf[a * N + j]) ** 2 for a in range(dim))) for i, j in pairs))
    wall = time.time() - t0
    t1 = time.time()
    chk_arith = get_arith(f"mpfr-{p['check_bits']}")
    chk_arith.activate()
    chk = TaylorStepper(case, p["check_order"], chk_arith)
    zc = [chk_arith.convert(v) for v in z0]
    idx = case.system.positions
    max_diff, diff_T = 0.0, 0.0
    for k, h in enumerate(steps):
        zc = chk.horner(chk.coeffs(zc), chk_arith.convert(h))
        ref_k = _load(states[k + 1])
        diff_T = error_positions(chk_arith, zc, ref_k, idx)
        max_diff = max(max_diff, diff_T)
    wall_check = time.time() - t1
    arith.activate()
    payload = {"steps": steps, "states": states, "grid_times": grid_times, "grid_states": [grid_states[i] for i in range(len(grid_times))],
               "rho": rhos}
    tmp = d / f"ref.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=5)
    tmp.replace(d / "ref.pkl")
    hs = [float(h) for h in steps]
    meta = {"case": case.id, "ic_id": ic_id, "protocol": protocol(case), "steps": len(steps), "hmin": min(hs), "hmax": max(hs),
            "dmin": dmin, "max_pos_diff": max_diff, "diff_at_T": diff_T, "tol_check": p.get("tol_check"),
            "check_passed": bool(p.get("tol_check") is None or max_diff < float(p["tol_check"])),
            "wall_reference": wall, "wall_check": wall_check, "ic": [str(x) for x in z0], "provenance": provenance()}
    _atomic_json(d / "meta.json", meta)
    return meta


class Reference:
    def __init__(self, case, ic_id: int):
        d = ref_dir(case, ic_id)
        if not (d / "meta.json").exists():
            compute_reference(case.id, ic_id)
        with open(d / "ref.pkl", "rb") as f:
            self.raw = pickle.load(f)
        self.meta = json.loads((d / "meta.json").read_text())
        if not self.meta.get("check_passed", True):
            raise RuntimeError(f"reference {case.id} #{ic_id} failed its check: max diff {self.meta['max_pos_diff']:.3g} "
                               f"against tol {self.meta['tol_check']}")
        self.steps = self.raw["steps"]
        self.grid_times = self.raw["grid_times"]
        idx = case.system.positions
        self.idx = idx
        self._grid = None
        self._pos = None
        self.times = None

    def pos(self):
        """Per-step reference, padded so that index i is the state component i of the case's position list."""
        if self._pos is None:
            self._pos = [_sparse(s, self.idx) for s in self.raw["states"]]
        return self._pos

    def grid(self):
        if self._grid is None:
            self._grid = [_sparse(s, self.idx) for s in self.raw["grid_states"]]
        return self._grid

    def full_state(self, k: int):
        return _load(self.raw["states"][k])

    def full_grid_state(self, k: int):
        return _load(self.raw["grid_states"][k])

    def step_times(self):
        if self.times is None:
            t, out = Fraction(0), [0.0]
            for h in self.steps:
                t += h
                out.append(float(t))
            self.times = out
        return self.times


def _sparse(state_bin, idx):
    out = [None] * (max(idx) + 1)
    for i in idx:
        out[i] = gmpy2.from_binary(state_bin[i])
    return out


def get_reference(case, ic_id: int) -> Reference:
    key = (case.id, ic_id)
    if key not in _loaded:
        while len(_loaded) >= 2:  # queue order keeps one IC's jobs together; each reference can be hundreds of MB
            _loaded.pop(next(iter(_loaded)))
        _loaded[key] = Reference(case, ic_id)
    return _loaded[key]
