"""Lie-series coefficients L_f^m x / m! in exact rationals, the validation rule of spec §4.2."""
from __future__ import annotations

from fractions import Fraction
from math import factorial

import sympy as sp

from .poly import PolySystem


def lie_coefficients(system: PolySystem, z0: list, order: int, variables=None) -> dict:
    """{state index: [coefficients 0..order]} from sparse-polynomial Lie derivatives, evaluated at z0."""
    names = [str(s) for s in system.states]
    R = sp.ring(names, sp.QQ)[0]
    gens = R.gens
    to_ring = dict(zip(system.states, gens))

    def conv(expr):
        return _expr_to_ring(sp.sympify(expr), to_ring, R)

    f = [conv(system.rhs[s]) for s in system.states]
    point = [sp.QQ(z.numerator, z.denominator) for z in z0]
    out = {}
    for idx in (variables if variables is not None else range(len(gens))):
        g = gens[idx]
        coeffs = []
        for m in range(order + 1):
            val = g.evaluate(list(zip(gens, point))) if g != 0 else sp.QQ(0)
            coeffs.append(Fraction(int(sp.QQ.numer(val)), int(sp.QQ.denom(val))) / factorial(m))
            if m < order:
                g = _lie_ring(g, f, gens)
        out[idx] = coeffs
    return out


def _lie_ring(g, f, gens):
    res = 0
    for gi, fi in zip(gens, f):
        d = g.diff(gi)
        if d != 0 and fi != 0:
            res = res + d * fi
    return res if res != 0 else g.ring.zero


def _expr_to_ring(e, to_ring, R):
    if e.is_Symbol:
        return to_ring[e]
    if e.is_Rational:
        return R(sp.QQ(int(e.p), int(e.q)))
    if e.is_Add:
        out = R.zero
        for a in e.args:
            out = out + _expr_to_ring(a, to_ring, R)
        return out
    if e.is_Mul:
        out = R.one
        for a in e.args:
            out = out * _expr_to_ring(a, to_ring, R)
        return out
    if e.is_Pow and e.exp.is_Integer and e.exp > 0:
        return _expr_to_ring(e.base, to_ring, R) ** int(e.exp)
    raise ValueError(f"not polynomial: {e}")
