"""Gravitational N-body in lifted polynomial form (spec §4.1, §6.1): u_ij = 1/r_ij, degree-5 RHS, G = 1."""
from __future__ import annotations

from itertools import combinations

import sympy as sp

from ..symbolic.poly import PolySystem


def build(name: str, masses: list, dim: int = 2) -> PolySystem:
    """State order: q per axis, v per axis, then u for pairs (i<j); matches pilot/taylor3b.py for N = 3, dim = 2."""
    N = len(masses)
    axes = "xyz"[:dim]
    q = {a: sp.symbols(f"q{a}0:{N}", real=True) for a in axes}
    v = {a: sp.symbols(f"v{a}0:{N}", real=True) for a in axes}
    pairs = list(combinations(range(N), 2))
    u = {p: sp.Symbol(f"u{p[0]}_{p[1]}", real=True) for p in pairs}
    uij = lambda i, j: u[(min(i, j), max(i, j))]  # noqa: E731
    m = [sp.Rational(x.numerator, x.denominator) for x in masses]
    rhs = {}
    states = [q[a][i] for a in axes for i in range(N)] + [v[a][i] for a in axes for i in range(N)] + [u[p] for p in pairs]
    for a in axes:
        for i in range(N):
            rhs[q[a][i]] = v[a][i]
            rhs[v[a][i]] = sum(m[j] * (q[a][j] - q[a][i]) * uij(i, j) ** 3 for j in range(N) if j != i)
    constraints, aux = {}, {}
    for i, j in pairs:
        s = sum((q[a][i] - q[a][j]) * (v[a][i] - v[a][j]) for a in axes)
        rhs[u[(i, j)]] = -u[(i, j)] ** 3 * s
        r2 = sum((q[a][i] - q[a][j]) ** 2 for a in axes)
        constraints[f"C{i}_{j}"] = u[(i, j)] ** 2 * r2 - 1
        aux[u[(i, j)]] = 1 / sp.sqrt(r2)
    kin = sum(m[i] * sum(v[a][i] ** 2 for a in axes) for i in range(N)) * sp.Rational(1, 2)
    invariants = {"energy": kin - sum(m[i] * m[j] * u[(i, j)] for i, j in pairs)}
    for a in axes:
        invariants[f"P{a}"] = sum(m[i] * v[a][i] for i in range(N))
    if dim == 2:
        invariants["Lz"] = sum(m[i] * (q["x"][i] * v["y"][i] - q["y"][i] * v["x"][i]) for i in range(N))
    base = [q[a][i] for a in axes for i in range(N)] + [v[a][i] for a in axes for i in range(N)]
    qs = [q[a][i] for a in axes for i in range(N)]
    vs = [v[a][i] for a in axes for i in range(N)]
    accel = []
    for a in axes:
        for i in range(N):
            accel.append(sum(m[j] * (q[a][j] - q[a][i]) * _r2(q, axes, i, j) ** sp.Rational(-3, 2) for j in range(N) if j != i))
    aux_values = [_r2(q, axes, i, j) ** sp.Rational(-1, 2) for i, j in pairs]
    meta = {"N": N, "dim": dim, "pairs": pairs, "masses": masses,
            "separable": {"q": qs, "v": vs, "accel": accel, "aux": [u[p] for p in pairs], "aux_values": aux_values}}
    return PolySystem(name, states, rhs, invariants=invariants, constraints=constraints,
                      positions=list(range(dim * N)), aux_init=aux, base=base, meta=meta)


def _r2(q, axes, i, j):
    return sum((q[a][i] - q[a][j]) ** 2 for a in axes)
