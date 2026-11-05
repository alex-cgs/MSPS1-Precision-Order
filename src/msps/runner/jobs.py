"""Job kinds. Each takes a resolved config dict and returns (meta, data columns, state columns); runner stores them."""
from __future__ import annotations

import math
import pickle
import time
from fractions import Fraction

import gmpy2
import numpy as np

from .. import diagnostics as dg
from ..arith import get_arith
from ..integrators import make_stepper
from ..integrators.maps import IteratedMap, resolve_card
from ..integrators.taylor import TaylorStepper
from ..schedules import AdaptiveDyadic, FixedStep
from ..systems import load_case
from ..systems.ic import parse
from .references import compute_reference, get_reference, grid_dt

np.seterr(all="ignore")
BLOWUP = 1e8
N_OUTPUT_TIMES = 20  # state snapshots per run, kept for the confidence-region work (AGENT_PROMPT section 4 item 10)


def _output_indices(n_grid: int, count: int = N_OUTPUT_TIMES) -> set:
    return {max(1, round((i + 1) * n_grid / count)) - 1 for i in range(count)}


class _Rec:
    """Coarse E(t) series, invariants and output-time state snapshots for one run."""

    def __init__(self, case, arith, z0):
        self.case, self.arith = case, arith
        self.mon = dg.InvariantMonitor(case, arith, z0)
        self.cols = {c: [] for c in ["t", "E", "E_pair"] + self.mon.columns()}
        self.state_rows = {str(s): [] for s in case.system.states}
        self.state_t = []
        self.pairs = case.system.meta.get("pairs")

    def sample(self, t, E, z, ref_pos=None):
        self.cols["t"].append(t)
        self.cols["E"].append(E)
        self.cols["E_pair"].append(_pair_error(self.case, self.arith, z, ref_pos) if self.pairs and ref_pos else float("nan"))
        d = self.mon.deltas(z)
        for k in self.mon.columns():
            self.cols[k].append(d.get(k, float("nan")))

    def snapshot(self, t, z):
        self.state_t.append(t)
        f = self.arith.to_float
        for s, v in zip(self.case.system.states, z):
            self.state_rows[str(s)].append(f(v))

    def data(self):
        return self.cols

    def states(self):
        if not self.state_t:
            return None
        return {"t": self.state_t, **self.state_rows}


def _pair_error(case, arith, z, ref_pos) -> float:
    """Relative-distance E: max over pairs of |r_ij - r_ij_ref| (spec §4.7 N-body option), at diagnostic precision."""
    S = case.system
    N, dim = S.meta["N"], S.meta["dim"]
    try:
        with gmpy2.context(gmpy2.get_context(), precision=800):
            best = gmpy2.mpfr(0)
            for i, j in S.meta["pairs"]:
                r2, rr2 = gmpy2.mpfr(0), gmpy2.mpfr(0)
                for a in range(dim):
                    x = arith.to_mpfr(z[a * N + i]) - arith.to_mpfr(z[a * N + j])
                    xr = ref_pos[a * N + i] - ref_pos[a * N + j]
                    r2 += x * x
                    rr2 += xr * xr
                best = max(best, abs(gmpy2.sqrt(r2) - gmpy2.sqrt(rr2)))
            return float(best)
    except Exception:
        return math.inf


def _blown(arith, z, idx) -> bool:
    for i in idx:
        v = arith.to_float(z[i])
        if not math.isfinite(v) or abs(v) > BLOWUP:
            return True
    return False


def run_reference(cfg):
    meta = compute_reference(cfg["case"], cfg["ic"])
    if not meta["check_passed"]:
        raise RuntimeError(f"reference check failed: max diff {meta['max_pos_diff']:.3g} against tol {meta['tol_check']}")
    return ({"reference": {k: meta[k] for k in ("steps", "hmin", "hmax", "dmin", "max_pos_diff", "diff_at_T", "check_passed",
                                                "wall_reference", "wall_check")}}, None, None)


def run_traj(cfg):
    """One (n, p) run against the reference. schedule: shared | adaptive | fixed | embedded (spec §7.1)."""
    case = load_case(cfg["case"])
    arith = get_arith(cfg["arith"])
    arith.activate()
    ic = cfg["ic"]
    z0 = case.ic(ic)
    ref = get_reference(case, ic)
    arith.activate()
    stepper = make_stepper(cfg, case, arith)
    idx = case.system.positions
    T = Fraction(str(case.card["T"]))
    tols = cfg.get("tolerances", case.card["tolerances"])
    tracker = dg.TpredTracker(tols, float(T))
    stop_err = float(cfg.get("stop_err", 1.0))
    max_wall = float(cfg.get("max_wall", 1e9))
    z = [arith.convert(v) for v in z0]
    rec = _Rec(case, arith, z)
    outs = _output_indices(len(ref.grid_times))
    mode = cfg["schedule"]
    t0 = time.perf_counter()
    status = "completed"
    n_steps = 0

    grid_ref = ref.grid()
    if mode == "shared":
        m = int(cfg.get("substeps", 1))
        sample_all = bool(cfg.get("sample_all", False))  # full per-step E(t), only when asked (AGENT_PROMPT section 4 item 10)
        pos = ref.pos()
        hs = [arith.convert(h / m) for h in ref.steps]
        times = ref.step_times()
        dt = float(grid_dt(T, case.card["reference"].get("grid_dt")))
        next_out, gi = dt, 0
        n_grid = len(ref.grid_times)
        for k, h in enumerate(hs):
            for _ in range(m):
                z = stepper.step(z, h)
            n_steps += m
            t = times[k + 1]
            E = dg.error_positions(arith, z, pos[k + 1], idx)
            tracker.update(t, E)
            if sample_all or t >= next_out - 1e-12 or k == len(hs) - 1:
                rec.sample(t, E, z, pos[k + 1])
                while gi < n_grid and float(ref.grid_times[gi]) <= t + 1e-12:
                    if gi in outs:
                        rec.snapshot(t, z)
                    gi += 1
                next_out = dt * (math.floor(t / dt + 1e-12) + 1)
            if E > stop_err or _blown(arith, z, idx):
                status = "stopped_err" if E <= BLOWUP and math.isfinite(E) else "blown_up"
                break
            if k % 256 == 0 and time.perf_counter() - t0 > max_wall:
                status = "wall_budget"
                break
    else:
        grid = ref.grid_times
        gi, t = 0, Fraction(0)
        if mode == "adaptive":
            sched = AdaptiveDyadic(float(cfg.get("safety", 8)), Fraction(str(cfg.get("cap", "1/2"))))
        elif mode == "fixed":
            sched = FixedStep(Fraction(str(cfg["h"])))
        elif mode == "embedded":
            sched = None
        else:
            raise ValueError(mode)
        hmin = Fraction(1, 2**120)
        taylor = isinstance(stepper, TaylorStepper)
        h_em = Fraction(1, 64)
        rtol, atol = float(cfg.get("rtol", 1e-12)), float(cfg.get("atol", 1e-14))
        done = False
        while t < T and not done:
            if mode == "embedded":
                h = min(h_em, grid[gi] - t, T - t)
                scale = atol + rtol * np.abs([arith.to_float(v) for v in z])
                zn, err = stepper.step_err(z, arith.convert(h), scale)
                fac = 10.0 if err == 0 else min(5.0, max(0.2, 0.9 * float(err) ** (-1 / 8)))
                h_em = Fraction(2) ** math.floor(math.log2(float(h) * fac))
                if err > 1:
                    if h < hmin:
                        status = "step_underflow"
                        break
                    continue
                n_steps += 1
                z, t = zn, t + h
                c = None
            elif taylor:
                c = stepper.coeffs(z)
                h = min(sched.next_h(t, z, c, stepper), T - t)
                if h < hmin:
                    status = "step_underflow"
                    break
            else:
                c = None
                h = min(sched.next_h(t, z, None, stepper), grid[gi] - t, T - t)
            if mode != "embedded":
                while taylor and gi < len(grid) and grid[gi] <= t + h:
                    zg = stepper.horner(c, arith.convert(grid[gi] - t))
                    done |= _grid_sample(arith, zg, grid[gi], gi, grid_ref, idx, tracker, rec, outs, stop_err)
                    gi += 1
                if taylor:
                    z = arith.post_step(stepper.horner(c, arith.convert(h)))
                else:
                    z = stepper.step(z, arith.convert(h))
                t += h
                n_steps += 1
            if not taylor:
                while gi < len(grid) and grid[gi] <= t:
                    done |= _grid_sample(arith, z, grid[gi], gi, grid_ref, idx, tracker, rec, outs, stop_err)
                    gi += 1
            if _blown(arith, z, idx):
                status = "blown_up"
                break
            if done:
                status = "stopped_err"
            if n_steps % 256 == 0 and time.perf_counter() - t0 > max_wall:
                status = "wall_budget"
                break
    wall = time.perf_counter() - t0
    results = tracker.result()
    meta = {"status": status, "tpred": results, "steps": n_steps, "t_end": tracker.last_t, "wall": wall,
            "unit_roundoff": arith.unit_roundoff(), "ref_max_diff": ref.meta["max_pos_diff"],
            "ic": [str(x) for x in z0], "solver_iterations": getattr(stepper, "iterations", None),
            "solver_failures": getattr(stepper, "not_converged", None), "tape_mul": case.tape.n_mul}
    return meta, rec.data(), rec.states()


def _grid_sample(arith, z, tg, gi, grid_ref, idx, tracker, rec, outs, stop_err) -> bool:
    t = float(tg)
    E = dg.error_positions(arith, z, grid_ref[gi], idx)
    tracker.update(t, E)
    rec.sample(t, E, z, grid_ref[gi])
    if gi in outs:
        rec.snapshot(t, z)
    return E > stop_err


def run_local(cfg):
    """E1: one-step errors from >= 60 reference states at reference precision (spec §4.7)."""
    case = load_case(cfg["case"])
    ic = cfg["ic"]
    ref = get_reference(case, ic)
    bits = int(case.card["reference"]["bits"])
    arith = get_arith(f"mpfr-{bits}")
    arith.activate()
    n_states = int(cfg.get("states", 60))
    k_all = list(range(0, len(ref.steps), max(1, len(ref.steps) // n_states)))
    idx = case.system.positions
    rows = {"n": [], "k": [], "h": [], "rho": [], "log10_eps": []}
    for n in cfg["orders"]:
        st = TaylorStepper(case, n, arith)
        for k in k_all:
            z = ref.full_state(k)
            target = ref.full_state(k + 1)
            z1 = st.horner(st.coeffs(z), arith.convert(ref.steps[k]))
            e = dg.error_positions(arith, z1, target, idx)
            rows["n"].append(n)
            rows["k"].append(k)
            rows["h"].append(float(ref.steps[k]))
            rows["rho"].append(ref.raw["rho"][k])
            rows["log10_eps"].append(math.log10(max(e, 1e-300)))
    med = {n: float(np.median([v for m, v in zip(rows["n"], rows["log10_eps"]) if m == n])) for n in cfg["orders"]}
    return {"median": med, "n_states": len(k_all)}, rows, None


def perturb_index(case) -> int:
    """First non-position base state, or state 0 (pilot perturbs vx1 so position separation starts at zero)."""
    pos = set(case.system.positions)
    for s in case.system.base:
        i = case.system.states.index(s)
        if i not in pos:
            return i
    return 0


def run_amplification(cfg):
    """A(t): perturbed reference on the same steps, at reference order and precision (spec §4.7).

    Also a renormalized twin for cumulative log growth, valid past saturation; drives the FTLE model of E3."""
    case = load_case(cfg["case"])
    ic = cfg["ic"]
    ref = get_reference(case, ic)
    p = case.card["reference"]
    arith = get_arith(f"mpfr-{p['bits']}")
    arith.activate()
    eps = parse(_pert(cfg, case))
    i = perturb_index(case)
    z0 = list(case.ic(ic))
    z0[i] += eps
    st = TaylorStepper(case, p["order"], arith)
    z = [arith.convert(v) for v in z0]
    idx = case.system.positions
    times = ref.step_times()
    pos = ref.pos()
    seps = [0.0]
    d0 = arith.convert(eps)
    # renormalized twin: rescale the full-state separation back to eps at each step when it has grown 1e6-fold
    w = [arith.convert(v) for v in z0]
    logG = [0.0]
    cum = 0.0
    for k, h in enumerate(ref.steps):
        hc = arith.convert(h)
        z = st.horner(st.coeffs(z), hc)
        seps.append(dg.error_positions(arith, z, pos[k + 1], idx))
        w = st.horner(st.coeffs(w), hc)
        rk = ref.full_state(k + 1)
        d = max(abs(a - b) for a, b in zip(w, rk))
        g = d / d0
        if g > 1e6:
            w = [b + (a - b) / g for a, b in zip(w, rk)]
            cum += math.log(float(g))
            logG.append(cum)
        else:
            logG.append(cum + math.log(float(g)))
    pert = float(eps)
    A = dg.amplification(seps, pert)
    mask = np.asarray(times) > 5
    fit = float(np.polyfit(np.asarray(times)[mask], np.log(A[mask]), 1)[0]) if mask.sum() > 2 else float("nan")
    keep = _thin(len(times), int(cfg.get("max_rows", 20000)))
    data = {"t": np.asarray(times)[keep], "sep": np.asarray(seps)[keep], "A": A[keep], "log_growth": np.asarray(logG)[keep]}
    data["h"] = np.asarray([float(ref.steps[max(0, j - 1)]) for j in keep])
    data["rho"] = np.asarray([ref.raw["rho"][max(0, j - 1)] for j in keep])
    return {"perturbation": str(eps), "perturb_index": i, "A_T": float(A[-1]), "log10_A_T": float(np.log10(A[-1])),
            "lambda_fit": fit, "growth_T": float(logG[-1]), "lambda_cum": float(logG[-1]) / float(times[-1])}, data, None


def _pert(cfg, case) -> str:
    return str(cfg.get("perturbation", case.card.get("perturbation", "1/10000000000000000000000000000000000000000")))


def _thin(n: int, limit: int):
    return np.arange(n) if n <= limit else np.unique(np.linspace(0, n - 1, limit).astype(int))


def run_exact(cfg):
    """E4: exact rational bit growth over steps of size h from the case's reference schedule or a fixed dyadic h."""
    arith = get_arith("q")
    arith.activate()
    n = int(cfg["n"])
    budget = float(cfg.get("max_wall", 120))
    limit_bits = int(cfg.get("max_bits", 8_000_000))
    max_steps = int(cfg.get("max_steps", 6))
    t0 = time.perf_counter()
    case = load_case(cfg["case"])
    z0 = case.ic(cfg["ic"])
    if any(_bits(v) > 4096 for v in z0):
        return {"status": "skipped_irrational_ic"}, None, None
    st = TaylorStepper(case, n, arith)
    h_cfg = cfg.get("h")
    if h_cfg is not None:
        hs = [Fraction(str(h_cfg))] * max_steps
    else:
        hs = get_reference(case, cfg["ic"]).steps[:max_steps]
        arith.activate()
    z = [arith.convert(v) for v in z0]
    bits = [max(_qbits(arith, v) for v in z)]
    walls = [0.0]
    t = Fraction(0)
    for h in hs:
        z = st.horner(st.coeffs(z), arith.convert(h))
        t += h
        bits.append(max(_qbits(arith, v) for v in z))
        walls.append(time.perf_counter() - t0)
        if walls[-1] > budget or bits[-1] > limit_bits:
            break
    d = case.system.degree
    growth = [bits[i + 1] / bits[i] for i in range(len(bits) - 1)]
    return {"status": "completed", "degree": d, "predicted_factor": (d - 1) * n + 1, "growth": growth, "bits": bits,
            "walls": walls, "t_reached": float(t), "steps_done": len(bits) - 1,
            "ic": [str(x) for x in z0]}, {"step": list(range(len(bits))), "bits": bits, "wall": walls}, None


def _bits(q: Fraction) -> int:
    return q.numerator.bit_length() + q.denominator.bit_length()


def _qbits(arith, v) -> int:
    return int(v.p).bit_length() + int(v.q).bit_length()


def _fast_fraction_seed(seed, ic_id, bits):
    rng = np.random.default_rng([int(seed), int(ic_id)])
    return Fraction(int(rng.integers(1, 2**bits - 1)) | 1, 2**bits)


def run_map(cfg):
    """Maps: orbit in the chosen arithmetic vs a high-precision reference orbit; eventual period; lambda; exact growth."""
    spec = cfg["case"]
    card, params = resolve_card(spec)
    overrides = dict(cfg.get("params", {}))
    x0 = _fast_fraction_seed(card["ic"]["seed"], cfg["ic"], int(card["ic"]["bits"]))
    horizon = int(cfg.get("horizon", card.get("horizon", 2000)))
    tol = float(cfg.get("tol", 1e-6))
    use_ref = bool(cfg.get("reference", True))
    ref_orbit = []
    if use_ref:
        ref_bits = int(cfg.get("ref_bits", max(4096, 3 * horizon)))
        ra = get_arith(f"mpfr-{ref_bits}")
        ra.activate()
        rm = IteratedMap(spec, ra, overrides)
        xr = ra.convert(x0)
        for _ in range(horizon + 1):
            ref_orbit.append(xr)
            xr = rm.step(xr)
    arith = get_arith(cfg["arith"])
    arith.activate()
    mp = IteratedMap(spec, arith, overrides)
    x = arith.convert(x0)
    seen = {}
    k_cross, period, tail = None, None, None
    t0 = time.perf_counter()
    errs = []
    orbit_f = []
    keep_rows = int(cfg.get("keep_rows", 2000))
    for k in range(horizon + 1):
        fx = arith.to_float(x)
        if k <= keep_rows:
            orbit_f.append(fx)
        if use_ref:
            e = arith.diff_abs(x, ref_orbit[k])
            errs.append(e)
            if k_cross is None and e > tol:
                k_cross = k
        key = arith.to_fraction(x) if cfg["arith"].startswith(("q", "mpfr", "dd")) else fx
        if period is None:
            if key in seen:
                period = k - seen[key]
                tail = seen[key]
                if not use_ref:
                    break
            else:
                seen[key] = k
        x = mp.step(x)
    wall = time.perf_counter() - t0
    x_f = np.asarray(orbit_f)
    lam = _map_lyapunov(spec, x_f, params | overrides, card)
    meta = {"k_cross": k_cross, "censored": k_cross is None, "period": period, "transient": tail, "lambda_est": lam,
            "x0": str(x0), "wall": wall, "unit_roundoff": arith.unit_roundoff(), "horizon": horizon, "tol": tol}
    if not use_ref:
        return meta, {"x": x_f}, None
    m = min(len(errs), len(x_f))
    return meta, {"k": np.arange(m), "err": errs[:m], "x": x_f[:m]}, None


def run_exact_map(cfg):
    """E4 for maps: exact rational iterates until a bit or wall limit; growth vs degree per iteration."""
    spec = cfg["case"]
    card, _ = resolve_card(spec)
    overrides = dict(cfg.get("params", {}))
    arith = get_arith("q")
    arith.activate()
    mp = IteratedMap(spec, arith, overrides)
    x = arith.convert(_fast_fraction_seed(card["ic"]["seed"], cfg["ic"], int(card["ic"]["bits"])))
    bits = [_qbits(arith, x)]
    walls = [0.0]
    t0 = time.perf_counter()
    while walls[-1] < float(cfg.get("max_wall", 60)) and bits[-1] < int(cfg.get("max_bits", 4_000_000)) and len(bits) < int(cfg.get("max_steps", 40)):
        x = mp.step(x)
        bits.append(_qbits(arith, x))
        walls.append(time.perf_counter() - t0)
    return {"degree": mp.degree, "predicted_factor": mp.degree, "bits": bits, "walls": walls,
            "growth": [bits[i + 1] / bits[i] for i in range(len(bits) - 1)], "params": overrides}, {"step": list(range(len(bits))), "bits": bits, "wall": walls}, None


def run_kepler(cfg):
    """Two-body oracle: E(t) against Kepler's equation at grid times, no reference run (spec §10.2)."""
    import mpmath as mpm
    case = load_case(cfg["case"])
    arith = get_arith(cfg["arith"])
    arith.activate()
    st = TaylorStepper(case, int(cfg["n"]), arith)
    e = parse(case.params["e"])
    T = Fraction(str(case.card["T"]))
    dt = Fraction(str(cfg.get("dt", "1/4")))
    z = [arith.convert(v) for v in case.ic(0)]
    sched = AdaptiveDyadic(8, Fraction(1, 2))
    t, gt = Fraction(0), dt
    ts, Es = [], []
    t0 = time.perf_counter()
    with mpm.workdps(150):
        em = mpm.mpf(e.numerator) / e.denominator
        nmean = mpm.sqrt(1 - em**2)
        guess = None
        def exact(tq):
            nonlocal guess
            M = nmean * mpm.mpf(tq.numerator) / tq.denominator
            E_ = mpm.findroot(lambda x: x - em * mpm.sin(x) - M, guess if guess is not None else M)
            guess = E_ + (nmean * mpm.mpf(dt.numerator) / dt.denominator) / (1 - em * mpm.cos(E_))
            return mpm.cos(E_) - em, mpm.sqrt(1 - em**2) * mpm.sin(E_)
        while t < T:
            c = st.coeffs(z)
            h = min(sched.next_h(t, z, c, st), T - t)
            while gt <= t + h:
                zg = st.horner(c, arith.convert(gt - t))
                rx, ry = exact(gt)
                with gmpy2.context(gmpy2.get_context(), precision=500):
                    dx = arith.to_mpfr(zg[1]) - arith.to_mpfr(zg[0]) - gmpy2.mpfr(mpm.nstr(rx, 140))
                    dy = arith.to_mpfr(zg[3]) - arith.to_mpfr(zg[2]) - gmpy2.mpfr(mpm.nstr(ry, 140))
                    Es.append(float(max(abs(dx), abs(dy))))
                ts.append(float(gt))
                gt += dt
            z = st.horner(c, arith.convert(h))
            t += h
    return {"e": str(e), "E_T": Es[-1], "E_max": max(Es), "wall": time.perf_counter() - t0}, {"t": ts, "E": Es}, None


def run_closure(cfg):
    """Periodic-orbit oracle: |q(P) - q(0)| at a high-precision run, plus |v(P) - v(0)| (spec §10.2 figure-eight)."""
    case = load_case(cfg["case"])
    arith = get_arith(cfg["arith"])
    arith.activate()
    st = TaylorStepper(case, int(cfg["n"]), arith)
    P = parse(cfg["period"])
    z0 = case.ic(cfg["ic"])
    z = [arith.convert(v) for v in z0]
    t, sched = Fraction(0), AdaptiveDyadic(8, Fraction(1, 2))
    while t < P:
        c = st.coeffs(z)
        h = min(sched.next_h(t, z, c, st), P - t)
        z = st.horner(c, arith.convert(h))
        t += h
    d = [abs(float(arith.to_fraction(a) - b)) for a, b in zip(z, z0)]
    return {"max_diff": max(d), "pos_diff": max(d[i] for i in case.system.positions), "period": str(P)}, None, None


def _map_lyapunov(spec, x, params, card) -> float:
    kind = card["builder"]
    if kind == "logistic":
        d = float(parse(params["r"])) * (1 - 2 * x)
    elif kind == "tent":
        d = np.full_like(x, float(parse(params["mu"])))
    else:
        mu = float(parse(params["mu"]))
        w = params.get("w")
        if w is None:
            d = mu * np.pi * np.cos(np.pi * x)
        else:
            y = np.pi * x
            s = np.zeros_like(x)
            for j in range(int(w) + 1):
                s = s + (-1.0) ** j * y ** (2 * j) / math.factorial(2 * j)
            d = mu * np.pi * s
    with np.errstate(divide="ignore"):
        val = np.log(np.abs(d[np.isfinite(x)]))
    val = val[np.isfinite(val)]
    return float(val.mean()) if val.size else float("nan")


def run_map_diagram(cfg):
    """Lyapunov and attractor diagrams over mu (Kafetzis-style), vectorized float64/float32/float16."""
    dtype = {"f16": np.float16, "f32": np.float32, "f64": np.float64}[cfg["arith"]]
    mus = np.linspace(float(cfg["mu_min"]), float(cfg["mu_max"]), int(cfg["mu_points"]))
    w = cfg.get("w")
    kind = cfg.get("builder", "sine")
    x = np.full(mus.shape, 0.3, dtype=dtype)
    m = mus.astype(dtype)
    pi = dtype(np.pi)

    def f(x):
        if kind == "logistic":
            return m * x * (dtype(1) - x)
        if w is None:
            return m * np.sin(pi * x)
        y = pi * x
        acc = dtype((-1.0) ** int(w) / math.factorial(2 * int(w) + 1))
        for j in range(int(w) - 1, -1, -1):
            acc = acc * y * y + dtype((-1.0) ** j / math.factorial(2 * j + 1))
        return m * y * acc

    def df(x):
        xd = x.astype(np.float64)
        if kind == "logistic":
            return mus * (1 - 2 * xd)
        if w is None:
            return mus * np.pi * np.cos(np.pi * xd)
        y = np.pi * xd
        s = np.zeros_like(xd)
        for j in range(int(w) + 1):
            s = s + (-1.0) ** j * y ** (2 * j) / math.factorial(2 * j)
        return mus * np.pi * s

    for _ in range(int(cfg.get("transient", 2000))):
        x = f(x)
    lam = np.zeros_like(mus)
    N = int(cfg.get("iterations", 4000))
    samples = []
    for k in range(N):
        lam += np.log(np.maximum(np.abs(df(x)), 1e-300))
        if k >= N - int(cfg.get("keep", 120)):
            samples.append(x.astype(np.float64).copy())
        x = f(x)
    lam /= N
    samples = np.stack(samples, 1)
    distinct = np.array([len(np.unique(np.round(s[np.isfinite(s)], 9))) for s in samples])
    out = {"mu": mus, "lambda": lam, "distinct": distinct, "bounded": np.isfinite(samples).all(axis=1) & (np.abs(samples).max(axis=1) < 1e3)}
    for j in range(samples.shape[1]):
        out[f"x{j}"] = samples[:, j]
    return {"w": w, "arith": cfg["arith"]}, out, None


def run_stats(cfg):
    """E5: long per-run adaptive integration; histograms, lambda (two-trajectory, same arithmetic), escapes, T_pred.

    Checkpoints every checkpoint_s; `resume_token` file lives in the run directory (spec §9 rule 5)."""
    from ..io import run_dir
    case = load_case(cfg["case"])
    ic = cfg["ic"]
    arith = get_arith(cfg["arith"])
    arith.activate()
    z0 = case.ic(ic)
    n = int(cfg["n"])
    T = Fraction(str(cfg["T"]))
    T_ref = Fraction(str(case.card["T"]))
    ref = get_reference(case, ic) if cfg.get("track_tpred", True) else None
    arith.activate()
    st = TaylorStepper(case, n, arith)
    sched = AdaptiveDyadic(float(cfg.get("safety", 8)), Fraction(str(cfg.get("cap", "1/2"))))
    idx = case.system.positions
    obs = cfg["observables"]
    bins = {k: np.linspace(float(v[0]), float(v[1]), int(v[2]) + 1) for k, v in obs.items()}
    hist = {k: np.zeros(len(e) - 1, dtype=np.int64) for k, e in bins.items()}
    index = {str(s): i for i, s in enumerate(case.system.states)}
    sample_dt = Fraction(str(cfg["sample_dt"]))
    transient = Fraction(str(cfg.get("transient", 0)))
    renorm_dt = Fraction(str(cfg.get("renorm_dt", 1)))
    u = arith.unit_roundoff() or 1e-30
    d0 = max(1e-8, math.sqrt(u)) * 4
    d0c = arith.convert(Fraction(d0))
    i_p = perturb_index(case)
    z = [arith.convert(v) for v in z0]
    w = list(z)
    w[i_p] = w[i_p] + d0c
    tracker = dg.TpredTracker(case.card["tolerances"], float(T_ref))
    log_growth, t_lyap = 0.0, 0.0
    t = Fraction(0)
    next_sample = sample_dt
    next_renorm = renorm_dt
    gi = 0
    grid_ref = ref.grid() if ref else None
    grid = ref.grid_times if ref else []
    outs = _output_indices(N_OUTPUT_TIMES * 25)
    out_dt = T / N_OUTPUT_TIMES
    next_out = out_dt
    snap = {"t": []} | {str(s): [] for s in case.system.states}
    esc_time, esc_body, status, steps = None, None, "completed", 0
    three_body = case.system.meta.get("N") == 3
    S = case.system
    ckpt_path = run_dir(cfg["_study"], cfg["_run_id"]) / "checkpoint.pkl" if "_run_id" in cfg else None
    ckpt_s = float(cfg.get("checkpoint_s", 300))
    max_wall = float(cfg.get("max_wall", 1e9))
    wall0 = time.perf_counter()
    wall_prev = 0.0
    last_ckpt = wall0
    if ckpt_path is not None and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        z = [arith.convert(v) for v in ck["z"]]
        w = [arith.convert(v) for v in ck["w"]]
        t, next_sample, next_renorm, next_out, gi, log_growth, t_lyap, steps = (ck[k] for k in ("t", "next_sample", "next_renorm", "next_out", "gi", "log_growth", "t_lyap", "steps"))
        hist = ck["hist"]
        snap = ck["snap"]
        wall_prev = ck["wall"]
        tracker.hit, tracker.last_t = ck["tracker_hit"], ck["tracker_t"]
    hmin = Fraction(1, 2**100)
    ff = arith.to_float
    while t < T:
        c = st.coeffs(z)
        h = min(sched.next_h(t, z, c, st), T - t)
        if h < hmin:
            status = "step_underflow"
            break
        te = t + h
        while gi < len(grid) and grid[gi] <= te:
            zg = st.horner(c, arith.convert(grid[gi] - t))
            E = dg.error_positions(arith, zg, grid_ref[gi], idx)
            tracker.update(float(grid[gi]), E)
            gi += 1
        while next_sample <= te:
            zg = st.horner(c, arith.convert(next_sample - t))
            if next_sample > transient:
                _hist_add(hist, bins, zg, index, ff, S)
            next_sample += sample_dt
        while next_out <= te:
            zg = st.horner(c, arith.convert(next_out - t))
            snap["t"].append(float(next_out))
            for s, v in zip(S.states, zg):
                snap[str(s)].append(ff(v))
            next_out += out_dt
        hc = arith.convert(h)
        z = arith.post_step(st.horner(c, hc))
        w = arith.post_step(st.horner(st.coeffs(w), hc))
        t = te
        steps += 1
        if t >= next_renorm:
            d = max(abs(ff(a - b)) for a, b in zip(w, z))
            if d == 0 or not math.isfinite(d):
                w = list(z)
                w[i_p] = w[i_p] + d0c
                d = d0
            else:
                g = arith.convert(Fraction(d0 / d))
                w = [b + (a - b) * g for a, b in zip(w, z)]
            log_growth += math.log(d / d0)
            t_lyap = float(t)
            while next_renorm <= t:
                next_renorm += renorm_dt
        if _blown(arith, z, idx):
            status = "blown_up"
            break
        if three_body and steps % 16 == 0:
            b = dg.escape_state(S, [ff(v) for v in z[:12]], float(cfg.get("escape_r_factor", 10)))
            if b is not None:
                esc_time, esc_body = float(t), int(b)
                status = "escaped"
                break
        now = time.perf_counter()
        if ckpt_path is not None and now - last_ckpt > ckpt_s:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"z": [arith.to_fraction(v) for v in z], "w": [arith.to_fraction(v) for v in w], "t": t, "next_sample": next_sample,
                             "next_renorm": next_renorm, "next_out": next_out, "gi": gi, "log_growth": log_growth, "t_lyap": t_lyap, "steps": steps,
                             "hist": hist, "snap": snap, "wall": wall_prev + now - wall0, "tracker_hit": tracker.hit, "tracker_t": tracker.last_t}, f)
            tmp.replace(ckpt_path)
            last_ckpt = now
        if now - wall0 > max_wall:
            status = "wall_budget"
            break
    wall = wall_prev + time.perf_counter() - wall0
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
    lam = log_growth / t_lyap if t_lyap > 0 else float("nan")
    data = {}
    for k, counts in hist.items():
        data[f"{k}_count"] = counts
        data[f"{k}_left"] = bins[k][:-1]
        data[f"{k}_right"] = bins[k][1:]
    meta = {"status": status, "lambda_est": lam, "t_end": float(t), "steps": steps, "wall": wall, "tpred": tracker.result() if ref else None,
            "escape_time": esc_time, "escape_body": esc_body, "ic": [str(x) for x in z0], "unit_roundoff": arith.unit_roundoff(),
            "samples": {k: int(v.sum()) for k, v in hist.items()}}
    return meta, _pad(data), {"t": snap["t"], **{k: v for k, v in snap.items() if k != "t"}} if snap["t"] else None


def _pad(d):
    m = max(len(v) for v in d.values())
    return {k: np.concatenate([np.asarray(v, float), np.full(m - len(v), np.nan)]) for k, v in d.items()}


def _hist_add(hist, bins, z, index, ff, S):
    for k, e in bins.items():
        if k in index:
            val = ff(z[index[k]])
        elif k == "log10_rmin":
            N, dim = S.meta["N"], S.meta["dim"]
            zf = [ff(v) for v in z[:dim * N]]
            val = math.log10(max(min(dg.pair_distances(S, zf)), 1e-300))
        elif k == "rmax":
            N, dim = S.meta["N"], S.meta["dim"]
            zf = [ff(v) for v in z[:dim * N]]
            val = max(dg.pair_distances(S, zf))
        else:
            raise KeyError(k)
        j = np.searchsorted(e, val) - 1
        if 0 <= j < len(e) - 1:
            hist[k][j] += 1


def run_poincare(cfg):
    """E6/E7: Hénon–Heiles section x = 0, px > 0, from dense Taylor output, per (n, p) at fixed h or adaptive."""
    case = load_case(cfg["case"])
    arith = get_arith(cfg["arith"])
    arith.activate()
    z0 = [arith.convert(parse(v)) for v in cfg["z0"]] if "z0" in cfg else [arith.convert(v) for v in case.ic(cfg["ic"])]
    n, T = int(cfg["n"]), Fraction(str(cfg["T"]))
    st = TaylorStepper(case, n, arith)
    sched = FixedStep(Fraction(str(cfg["h"]))) if "h" in cfg else AdaptiveDyadic(8, Fraction(1, 2))
    i_x, i_y, i_px, i_py = [case.system.states.index(s) for s in case.system.states[:4]]
    z, t = z0, Fraction(0)
    ff = arith.to_float
    ys, pys, ts = [], [], []
    H0 = dg.InvariantMonitor(case, arith, z)
    prev_x = ff(z[i_x])
    t0 = time.perf_counter()
    maxdH = 0.0
    status = "completed"
    steps = 0
    while t < T:
        c = st.coeffs(z)
        h = sched.next_h(t, z, c, st)
        z_new = arith.post_step(st.horner(c, arith.convert(h)))
        xn = ff(z_new[i_x])
        if prev_x < 0 <= xn:
            # Newton on the Taylor polynomial of x(s) for the crossing, in float64 on float coefficients
            cx = [ff(v) for v in c[i_x]]
            s = float(h) * (-prev_x) / (xn - prev_x)
            for _ in range(4):
                p = np.polyval(cx[::-1], s)
                dp = np.polyval([k * cx[k] for k in range(len(cx) - 1, 0, -1)], s)
                if dp != 0:
                    s -= p / dp
            zs = st.horner(c, arith.convert(Fraction(s).limit_denominator(2**60)))
            ys.append(ff(zs[i_y]))
            pys.append(ff(zs[i_py]))
            ts.append(float(t) + s)
        prev_x = xn
        z, t = z_new, t + h
        steps += 1
        if steps % 64 == 0:
            maxdH = max(maxdH, H0.deltas(z).get("denergy", 0.0))
        if _blown(arith, z, range(2)):
            status = "blown_up"
            break
    meta = {"status": status, "crossings": len(ys), "max_dH": maxdH, "wall": time.perf_counter() - t0, "steps": steps}
    return meta, {"t": ts, "y": ys, "py": pys}, None


def run_strobe(cfg):
    """E6: Duffing stroboscopic section x(k P) per (gamma, n, p); h = P/m exact rational; period detection, lambda."""
    spec = cfg["case"]
    case = load_case(spec)
    arith = get_arith(cfg["arith"])
    arith.activate()
    omega = parse(case.params["omega"])
    # P rational to 40 digits; the forcing phase error over the run is far below any tolerance used here
    import mpmath as mp
    with mp.workdps(60):
        P = Fraction(int(mp.floor(2 * mp.pi / mp.mpf(omega.numerator) * omega.denominator * mp.mpf(10) ** 40)), 10**40)
    m = int(cfg.get("steps_per_period", 64))
    h = arith.convert(P / m)
    st = TaylorStepper(case, int(cfg["n"]), arith)
    z0 = case.ic(cfg["ic"])
    z = [arith.convert(v) for v in z0]
    ff = arith.to_float
    w = list(z)
    u = arith.unit_roundoff() or 1e-30
    d0 = max(1e-8, math.sqrt(u)) * 4
    d0c = arith.convert(Fraction(d0))
    w[1] = w[1] + d0c
    trans = int(cfg.get("transient_periods", 400))
    N = int(cfg.get("periods", 400))
    xs, vs = [], []
    log_growth = 0.0
    t0 = time.perf_counter()
    status = "completed"
    for k in range(trans + N):
        for _ in range(m):
            z = st.step(z, h)
            w = st.step(w, h)
        d = max(abs(ff(a - b)) for a, b in zip(w, z))
        if d == 0 or not math.isfinite(d):
            w = list(z)
            w[1] = w[1] + d0c
            d = d0
        else:
            g = arith.convert(Fraction(d0 / d))
            w = [b + (a - b) * g for a, b in zip(w, z)]
        if k >= trans:
            log_growth += math.log(d / d0)
            xs.append(ff(z[0]))
            vs.append(ff(z[1]))
        if _blown(arith, z, [0]):
            status = "blown_up"
            break
    xs = np.asarray(xs)
    lam = log_growth / (len(xs) * float(P)) if len(xs) else float("nan")
    return {"status": status, "lambda_est": lam, "distinct_1e-6": _distinct(xs, 1e-6), "wall": time.perf_counter() - t0,
            "P": str(P)}, {"x": xs, "v": np.asarray(vs)}, None


def _distinct(x, tol) -> int:
    x = np.sort(x[np.isfinite(x)])
    return 1 + int((np.diff(x) > tol).sum()) if x.size else 0


def run_indicator(cfg):
    """E7: sensitivity to (n, p) on a grid point: log10|z(T; n1,p1) - z(T; n2,p2)| against FTLE from a two-trajectory run."""
    case = load_case(cfg["case"])
    E = parse(case.params["E"])
    y, py = parse(cfg["y"]), parse(cfg["py"])
    rad = 2 * E - py * py - y * y + Fraction(2, 3) * y * y * y
    if rad <= 0:
        return {"status": "outside_energy_curve"}, None, None
    from ..symbolic.poly import _to_fraction
    import sympy as sp
    px = _to_fraction(sp.sqrt(sp.Rational(rad.numerator, rad.denominator)))
    z0 = [Fraction(0), y, px, py]
    T = Fraction(str(cfg["T"]))
    finals = []
    t0 = time.perf_counter()
    for pair in cfg["pairs"]:
        arith = get_arith(pair["arith"])
        arith.activate()
        st = TaylorStepper(case, int(pair["n"]), arith)
        h = arith.convert(Fraction(str(cfg["h"])))
        z = [arith.convert(v) for v in z0]
        for _ in range(int(T / Fraction(str(cfg["h"])))):
            z = st.step(z, h)
        finals.append([arith.to_fraction(v) for v in z])
    diffs = [float(max(abs(a - b) for a, b in zip(finals[0][:2], f[:2]))) for f in finals[1:]]
    arith = get_arith("mpfr-160")
    arith.activate()
    st = TaylorStepper(case, 24, arith)
    h = arith.convert(Fraction(str(cfg["h"])))
    z = [arith.convert(v) for v in z0]
    d0 = Fraction(1, 10**20)
    w = list(z)
    w[3] = w[3] + arith.convert(d0)
    lg = 0.0
    ns = int(T / Fraction(str(cfg["h"])))
    renorm = max(1, ns // 200)
    for k in range(ns):
        z = st.step(z, h)
        w = st.step(w, h)
        if (k + 1) % renorm == 0:
            d = max(abs(a - b) for a, b in zip(w, z))
            g = d / arith.convert(d0)
            lg += math.log(float(g))
            w = [b + (a - b) / g for a, b in zip(w, z)]
    return {"status": "completed", "sensitivity": [math.log10(max(d, 1e-300)) for d in diffs], "ftle": lg / float(T),
            "y": float(y), "py": float(py), "wall": time.perf_counter() - t0}, None, None


KINDS = {"reference": run_reference, "traj": run_traj, "local": run_local, "amplification": run_amplification, "exact": run_exact,
         "map": run_map, "exact_map": run_exact_map, "kepler": run_kepler, "closure": run_closure, "map_diagram": run_map_diagram, "stats": run_stats, "poincare": run_poincare, "strobe": run_strobe,
         "indicator": run_indicator}
