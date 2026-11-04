"""Störmer–Verlet and Yoshida compositions for separable Hamiltonians (spec §7.1).

Operates on the original (q, v); lifted auxiliaries (u_ij) are recomputed from q for diagnostics. Yoshida 4 uses
exact algebraic coefficients; orders 6 and 8 use Yoshida's (1990) published 15-digit values (solution A of each table),
so their order conditions hold only to about 1e-15.
"""
from __future__ import annotations

from fractions import Fraction

import mpmath as mp

from ..symbolic import compile_values

_Y6 = ["-1.17767998417887", "0.235573213359357", "0.784513610477560"]
_Y8 = ["-0.161582374150097e1", "-0.244699182370524e1", "-0.716989419708120e-2", "0.244002732616735e1",
       "0.157739928123617e0", "0.182020630970714e1", "0.104242620869991e1"]


def composition(order: int) -> list:
    if order == 2:
        return [Fraction(1)]
    if order == 4:
        with mp.workdps(320):
            w1 = 1 / (2 - mp.cbrt(2))
            w0 = 1 - 2 * w1
            q = lambda x: Fraction(int(mp.floor(x * mp.mpf(10) ** 310)), 10**310)  # noqa: E731
            return [q(w1), q(w0), q(w1)]
    table = {6: _Y6, 8: _Y8}[order]
    w = [Fraction(x) for x in table]
    w0 = 1 - 2 * sum(w)
    return list(reversed(w)) + [w0] + w


class SplitStepper:
    name = "yoshida"

    def __init__(self, case, arith, order: int = 2):
        sep = case.system.meta.get("separable")
        if sep is None:
            raise ValueError(f"{case.id} is not separable")
        self.case, self.arith = case, arith
        self.order = order
        S = case.system
        self.iq = [S.states.index(s) for s in sep["q"]]
        self.iv = [S.states.index(s) for s in sep["v"]]
        self.iaux = [S.states.index(s) for s in sep["aux"]]
        self._accel, c = compile_values(sep["q"], sep["accel"], "accel")
        self.Ka = [arith.convert(x) for x in c]
        self._aux = None
        if sep["aux"]:
            self._aux, c2 = compile_values(sep["q"], sep["aux_values"], "aux")
            self.Kx = [arith.convert(x) for x in c2]
        self.w = [arith.convert(x) for x in composition(order)]
        self.half = arith.convert(Fraction(1, 2))
        self.evals_per_step = len(self.w)

    def _verlet(self, q, v, h):
        hh = h * self.half
        q = [a + hh * b for a, b in zip(q, v)]
        acc = self._accel(q, self.Ka, self.arith.sqrt)
        v = [a + h * b for a, b in zip(v, acc)]
        q = [a + hh * b for a, b in zip(q, v)]
        return q, v

    def step(self, z, h):
        q = [z[i] for i in self.iq]
        v = [z[i] for i in self.iv]
        for w in self.w:
            q, v = self._verlet(q, v, w * h)
        out = list(z)
        for i, x in zip(self.iq, q):
            out[i] = x
        for i, x in zip(self.iv, v):
            out[i] = x
        if self._aux is not None:
            for i, x in zip(self.iaux, self._aux(q, self.Kx, self.arith.sqrt)):
                out[i] = x
        return self.arith.post_step(out)
