"""Non-gravitational ODE case builders (spec §6.2). Parameters arrive from the YAML cards as Fractions."""
from __future__ import annotations

import sympy as sp

from ..symbolic.poly import PolySystem, polynomialize


def _r(q):
    return sp.Rational(q.numerator, q.denominator)


def lorenz63(name, p):
    x, y, z = sp.symbols("x y z", real=True)
    s, r, b = _r(p["sigma"]), _r(p["rho"]), _r(p["beta"])
    rhs = {x: s * (y - x), y: x * (r - z) - y, z: x * y - b * z}
    return PolySystem(name, [x, y, z], rhs, positions=[0, 1, 2], base=[x, y, z])


def rossler(name, p):
    x, y, z = sp.symbols("x y z", real=True)
    a, b, c = _r(p["a"]), _r(p["b"]), _r(p["c"])
    rhs = {x: -y - z, y: x + a * y, z: b + z * (x - c)}
    return PolySystem(name, [x, y, z], rhs, positions=[0, 1, 2], base=[x, y, z])


def henon_heiles(name, p):
    x, y, px, py = sp.symbols("x y px py", real=True)
    rhs = {x: px, y: py, px: -x - 2 * x * y, py: -y - x**2 + y**2}
    H = (px**2 + py**2) / 2 + (x**2 + y**2) / 2 + x**2 * y - y**3 / 3
    meta = {"separable": {"q": [x, y], "v": [px, py], "accel": [-x - 2 * x * y, -y - x**2 + y**2], "aux": [], "aux_values": []}}
    return PolySystem(name, [x, y, px, py], rhs, invariants={"energy": H}, positions=[0, 1], base=[x, y, px, py], meta=meta)


def double_pendulum(name, p):
    m1, m2, l1, l2, g = (_r(p[k]) for k in ("m1", "m2", "l1", "l2", "g"))
    t1, t2, w1, w2 = sp.symbols("theta1 theta2 omega1 omega2", real=True)
    d = t1 - t2
    den = 2 * m1 + m2 - m2 * sp.cos(2 * t1 - 2 * t2)
    a1 = (-g * (2 * m1 + m2) * sp.sin(t1) - m2 * g * sp.sin(t1 - 2 * t2)
          - 2 * sp.sin(d) * m2 * (w2**2 * l2 + w1**2 * l1 * sp.cos(d))) / (l1 * den)
    a2 = 2 * sp.sin(d) * (w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * sp.cos(t1) + w2**2 * l2 * m2 * sp.cos(d)) / (l2 * den)
    rhs = {t1: w1, t2: w2, w1: a1, w2: a2}
    E = ((m1 + m2) * l1**2 * w1**2 + m2 * l2**2 * w2**2) / 2 + m2 * l1 * l2 * w1 * w2 * sp.cos(d) \
        - (m1 + m2) * g * l1 * sp.cos(t1) - m2 * g * l2 * sp.cos(t2)
    return polynomialize(name, [t1, t2, w1, w2], rhs, positions=[0, 1], invariants={"energy": E})


def duffing(name, p):
    dl, om, ga = _r(p["delta"]), _r(p["omega"]), _r(p["gamma"])
    x, v, tau = sp.symbols("x v tau", real=True)
    rhs = {x: v, v: -dl * v + x - x**3 + ga * sp.cos(om * tau), tau: sp.Integer(1)}
    return polynomialize(name, [x, v, tau], rhs, positions=[0])


def lorenz96(name, p):
    N, F = int(p["N"]), _r(p["F"])
    x = sp.symbols(f"x0:{N}", real=True)
    rhs = {x[i]: (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F for i in range(N)}
    return PolySystem(name, list(x), rhs, positions=list(range(N)), base=list(x))


def fput_beta(name, p):
    """Fixed-end chain, N moving particles (spec §6.2)."""
    N, beta = int(p["N"]), _r(p["beta"])
    q = sp.symbols(f"q0:{N}", real=True)
    v = sp.symbols(f"v0:{N}", real=True)
    Q = lambda i: 0 if i < 0 or i >= N else q[i]  # noqa: E731
    accel = []
    for i in range(N):
        fwd, bwd = Q(i + 1) - Q(i), Q(i) - Q(i - 1)
        accel.append(fwd - bwd + beta * (fwd**3 - bwd**3))
    rhs = {q[i]: v[i] for i in range(N)} | {v[i]: accel[i] for i in range(N)}
    bonds = [Q(i + 1) - Q(i) for i in range(-1, N)]
    H = sum(w**2 for w in v) / 2 + sum(b**2 for b in bonds) / 2 + beta * sum(b**4 for b in bonds) / 4
    meta = {"separable": {"q": list(q), "v": list(v), "accel": accel, "aux": [], "aux_values": []}}
    return PolySystem(name, list(q) + list(v), rhs, invariants={"energy": H}, positions=list(range(N)),
                      base=list(q) + list(v), meta=meta)
