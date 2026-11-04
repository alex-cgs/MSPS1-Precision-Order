"""Seeded rational initial-condition generators. Every generator: (case, ic_id) -> {base Symbol: Fraction}."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import sympy as sp


def parse(v) -> Fraction:
    """Fraction from a YAML value; symbolic strings such as '3**(-1/4)' go through SymPy at IC_DIGITS digits."""
    try:
        return Fraction(str(v))
    except ValueError:
        from ..symbolic.poly import _to_fraction
        return _to_fraction(sp.sympify(str(v), rational=True))


def _rng(card, ic_id, attempt=0):
    return np.random.default_rng([int(card["ic"]["seed"]), int(ic_id), attempt])


def _dyadic(x: float, bits: int = 30) -> Fraction:
    return Fraction(round(x * 2**bits), 2**bits)


def _zero_momentum(masses, vel):
    M = sum(masses)
    for a in range(len(vel)):
        mean = sum(m * w for m, w in zip(masses, vel[a])) / M
        vel[a] = [w - mean for w in vel[a]]
    return vel


def _nbody_base(sys, pos, vel):
    """pos, vel: [axis][body] Fractions."""
    N, dim = sys.meta["N"], sys.meta["dim"]
    vals = [pos[a][i] for a in range(dim) for i in range(N)] + [vel[a][i] for a in range(dim) for i in range(N)]
    return dict(zip(sys.base, vals))


def fixed(case, ic_id):
    values = case.card["ic"]["values"]
    return {s: parse(values[str(s)]) for s in case.system.base}


def kepler(case, ic_id):
    """a = 1, m1 = m2 = (1-e^2)/2, so GM = 1-e^2 and the periapsis speed is exactly 1+e (spec §6.1 oracle)."""
    e = parse(case.params["e"])
    M = 1 - e * e
    rp = 1 - e
    vp = 1 + e
    # m1 = m2, so each body sits at half the relative separation
    pos = [[-rp / 2, rp / 2], [Fraction(0), Fraction(0)]]
    vel = [[Fraction(0), Fraction(0)], [-vp / 2, vp / 2]]
    assert sum(case.system.meta["masses"]) == M
    return _nbody_base(case.system, pos, vel)


def _circle_points(ts, R):
    """Rational points on a circle with rational mutual distances (spec §6.1)."""
    pts = []
    for t in ts:
        c, s = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
        pts.append((R * (c * c - s * s), R * 2 * s * c))
    return pts


def ensemble3(case, ic_id):
    """Bound, zero-momentum planar triples on a rational circle; filtered in Case.ic (spec §6.1 NB3-ensemble)."""
    ic = case.card["ic"]
    attempt = case._attempt
    rng = _rng(case.card, ic_id, attempt)
    ts = sorted(Fraction(int(a), 64) for a in rng.integers(-256, 257, size=3))
    R = parse(ic["R_min"]) + Fraction(int(rng.integers(0, 33)), 32) * (parse(ic["R_max"]) - parse(ic["R_min"]))
    pts = _circle_points(ts, R)
    vm = parse(ic["vmax"])
    vel = [[vm * Fraction(int(a), 64) for a in rng.integers(-64, 65, size=3)] for _ in range(2)]
    masses = case.system.meta["masses"]
    vel = _zero_momentum(masses, vel)
    pos = [[p[0] for p in pts], [p[1] for p in pts]]
    return _nbody_base(case.system, pos, vel)


def ring(case, ic_id):
    """N bodies, angles near-even via rational tan(alpha/2); speeds v0 +- seeded jitter; zero momentum."""
    ic = case.card["ic"]
    N = case.system.meta["N"]
    rng = _rng(case.card, ic_id)
    den = int(ic.get("t_den", 4096))
    ts = [Fraction(round(math.tan(math.pi * k / (2 * N)) * den), den) for k in range(N)]
    R = parse(ic["R"])
    pts = _circle_points(ts, R)
    v0, jitter = parse(ic["v0"]), parse(ic["jitter"])
    vel = [[], []]
    for k, (t, (x, y)) in enumerate(zip(ts, pts)):
        cx, sx = x / R, y / R
        f = 1 + jitter * Fraction(int(rng.integers(-64, 65)), 64)
        vel[0].append(-v0 * f * sx)
        vel[1].append(v0 * f * cx)
    vel = _zero_momentum(case.system.meta["masses"], vel)
    return _nbody_base(case.system, [[p[0] for p in pts], [p[1] for p in pts]], vel)


def two_rings(case, ic_id):
    """Concentric rings with alternating bodies; cross distances are irrational, so u(0) at IC_DIGITS (spec §6.1)."""
    ic = case.card["ic"]
    N = case.system.meta["N"]
    rng = _rng(case.card, ic_id)
    den = 4096
    R1, R2 = parse(ic["R1"]), parse(ic["R2"])
    v0, jitter = parse(ic["v0"]), parse(ic["jitter"])
    pos, vel = [[], []], [[], []]
    for k in range(N):
        t = Fraction(round(math.tan(math.pi * k / (2 * N)) * den), den)
        R = R1 if k % 2 == 0 else R2
        (x, y), = _circle_points([t], R)
        cx, sx = x / R, y / R
        f = 1 + jitter * Fraction(int(rng.integers(-64, 65)), 64)
        pos[0].append(x); pos[1].append(y)
        vel[0].append(-v0 * f * sx); vel[1].append(v0 * f * cx)
    vel = _zero_momentum(case.system.meta["masses"], vel)
    return _nbody_base(case.system, pos, vel)


def hierarchical(case, ic_id):
    """Tight binary (bodies 0, 1) on a circular relative orbit, other bodies on a wide perturber ring."""
    ic = case.card["ic"]
    masses = case.system.meta["masses"]
    N = len(masses)
    rng = _rng(case.card, ic_id)
    a, Rp, vp = parse(ic["a"]), parse(ic["R"]), parse(ic["v0"])
    mb = masses[0] + masses[1]
    vb = _dyadic(float(mb / a) ** 0.5)  # rational approximation of the circular speed
    pos = [[-a * masses[1] / mb, a * masses[0] / mb], [Fraction(0), Fraction(0)]]
    vel = [[Fraction(0), Fraction(0)], [-vb * masses[1] / mb, vb * masses[0] / mb]]
    den = 4096
    for k in range(2, N):
        t = Fraction(round(math.tan(math.pi * (k - 2) / (2 * (N - 2))) * den), den)
        (x, y), = _circle_points([t], Rp)
        f = 1 + parse(ic["jitter"]) * Fraction(int(rng.integers(-64, 65)), 64)
        pos[0].append(x); pos[1].append(y)
        vel[0].append(-vp * f * y / Rp); vel[1].append(vp * f * x / Rp)
    vel = _zero_momentum(masses, vel)
    return _nbody_base(case.system, pos, vel)


def plummer(case, ic_id):
    """Plummer-like cluster, float-generated then converted to rationals; velocities virial-scaled (spec §6.1)."""
    ic = case.card["ic"]
    masses = case.system.meta["masses"]
    N, dim = len(masses), case.system.meta["dim"]
    rng = _rng(case.card, ic_id)
    a = float(parse(ic["scale"]))
    x = rng.uniform(1e-3, 0.9, size=N)
    r = a * (x ** (-2 / 3) - 1) ** -0.5
    ang = rng.uniform(0, 2 * math.pi, size=N)
    q = np.stack([r * np.cos(ang), r * np.sin(ang)])
    v = rng.normal(size=(dim, N))
    m = np.array([float(w) for w in masses])
    q -= (q * m).sum(axis=1, keepdims=True) / m.sum()
    v -= (v * m).sum(axis=1, keepdims=True) / m.sum()
    pot = sum(m[i] * m[j] / np.linalg.norm(q[:, i] - q[:, j]) for i in range(N) for j in range(i + 1, N))
    kin = 0.5 * (m * (v**2).sum(axis=0)).sum()
    v *= math.sqrt(float(parse(ic["virial"])) * pot / (2 * kin)) if kin > 0 else 1.0
    pos = [[_dyadic(float(w)) for w in q[a_]] for a_ in range(dim)]
    vel = _zero_momentum(masses, [[_dyadic(float(w)) for w in v[a_]] for a_ in range(dim)])
    return _nbody_base(case.system, pos, vel)


def box(case, ic_id):
    """Uniform seeded dyadic rationals in `box: {name: [lo, hi]}`; missing names take `values`."""
    ic = case.card["ic"]
    rng = _rng(case.card, ic_id)
    out = {}
    for s in case.system.base:
        key = str(s)
        rng_box = ic.get("box", {}).get(key, ic.get("default_box"))
        if rng_box is not None:
            lo, hi = map(float, rng_box)
            out[s] = _dyadic(rng.uniform(lo, hi), int(ic.get("bits", 30)))
        else:
            out[s] = parse(ic["values"][key])
    return out


def attractor(case, ic_id):
    """Box point carried for `transient` time units at 200 bits, then rounded to dyadic rationals (spec: on the attractor)."""
    from ..arith import get_arith
    from ..integrators.taylor import TaylorStepper
    from ..schedules import AdaptiveDyadic
    ic = case.card["ic"]
    start = box(case, ic_id)
    z = [start[s] for s in case.system.states]
    arith = get_arith("mpfr-200")
    arith.activate()
    st = TaylorStepper(case, 30, arith)
    sched = AdaptiveDyadic(safety=8, cap=Fraction(1, 2))
    Z = [arith.convert(v) for v in z]
    t, T = Fraction(0), parse(ic["transient"])
    while t < T:
        c = st.coeffs(Z)
        h = min(sched.next_h(t, Z, c, st), T - t)
        Z = st.horner(c, arith.convert(h))
        t += h
    bits = int(ic.get("bits", 40))
    vals = [_dyadic(float(arith.to_fraction(v)), bits) for v in Z]
    return dict(zip(case.system.states, vals))


def hh_shell(case, ic_id):
    """x = 0, seeded (y, py) inside the energy curve, px > 0 from H = E at IC_DIGITS digits."""
    ic = case.card["ic"]
    E = parse(case.params["E"])
    for attempt in range(10000):
        rng = _rng(case.card, ic_id, attempt)
        y = _dyadic(rng.uniform(-0.6, 1.0), 20)
        py = _dyadic(rng.uniform(-0.6, 0.6), 20)
        rad = 2 * E - py * py - y * y + Fraction(2, 3) * y * y * y
        if rad > 0 and abs(y) < 1:
            from ..symbolic.poly import _to_fraction
            px = _to_fraction(sp.sqrt(sp.Rational(rad.numerator, rad.denominator)))
            vals = {"x": Fraction(0), "y": y, "px": px, "py": py}
            return {s: vals[str(s)] for s in case.system.base}
    raise RuntimeError("no point inside the energy curve")


def fput_thermal(case, ic_id):
    """q = 0, seeded dyadic velocities scaled to energy density `e`, zero mean is not imposed (fixed ends)."""
    ic = case.card["ic"]
    N = len(case.system.base) // 2
    rng = _rng(case.card, ic_id)
    v = rng.normal(size=N)
    v *= math.sqrt(2 * float(parse(case.params["e"])) * N / (v**2).sum())
    vals = {f"q{i}": Fraction(0) for i in range(N)} | {f"v{i}": _dyadic(float(v[i]), 30) for i in range(N)}
    return {s: vals[str(s)] for s in case.system.base}


GENERATORS = {f.__name__: f for f in (fixed, kepler, ensemble3, ring, two_rings, hierarchical, plummer, box, attractor,
                                      hh_shell, fput_thermal)}
