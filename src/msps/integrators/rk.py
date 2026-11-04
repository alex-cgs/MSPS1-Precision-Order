"""Explicit Runge–Kutta methods written over any arithmetic.

DOP853 tableau: SciPy's Hairer coefficients are float64, converted exactly to Fractions. That limits the method
above 53 bits (order conditions hold only to about 1e-17): E8 runs it in f64 and dd, and the limit is recorded there.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


class _RHS:
    def __init__(self, case, arith):
        fn, consts = case.value_fn("rhs")
        self.fn = fn
        self.K = [arith.convert(c) for c in consts]
        self.arith = arith

    def __call__(self, z):
        return self.fn(z, self.K)


class RK4:
    name = "rk4"
    order = 4
    evals_per_step = 4

    def __init__(self, case, arith):
        self.f = _RHS(case, arith)
        self.arith = arith
        c = arith.convert
        self.half, self.sixth, self.third = c(Fraction(1, 2)), c(Fraction(1, 6)), c(Fraction(1, 3))

    def step(self, z, h):
        f, half = self.f, self.half
        hh = h * half
        k1 = f(z)
        k2 = f([a + hh * b for a, b in zip(z, k1)])
        k3 = f([a + hh * b for a, b in zip(z, k2)])
        k4 = f([a + h * b for a, b in zip(z, k3)])
        s, t = h * self.sixth, h * self.third
        return self.arith.post_step([a + s * (b + d) + t * (e + g) for a, b, e, g, d in zip(z, k1, k2, k3, k4)])


class DOP853:
    """Dormand–Prince 8(5,3) with Hairer's combined error estimate; `step` gives one fixed step of order 8."""
    name = "dop853"
    order = 8
    evals_per_step = 12

    def __init__(self, case, arith):
        from scipy.integrate._ivp import dop853_coefficients as dc
        self.f = _RHS(case, arith)
        self.arith = arith
        c = lambda x: arith.convert(Fraction(float(x)))  # noqa: E731
        s = dc.N_STAGES
        self.A = [[(j, c(dc.A[i, j])) for j in range(i) if dc.A[i, j] != 0.0] for i in range(s)]
        self.B = [(i, c(dc.B[i])) for i in range(s) if dc.B[i] != 0.0]
        self.E5 = [float(x) for x in dc.E5[:s]]
        self.E3 = [float(x) for x in dc.E3[:s]]
        self.E5_extra = float(dc.E5[s]) if len(dc.E5) > s else 0.0
        self.E3_extra = float(dc.E3[s]) if len(dc.E3) > s else 0.0
        self.s = s
        self.last_err = 0.0

    def _stages(self, z, h):
        k = [self.f(z)]
        for i in range(1, self.s):
            zi = list(z)
            for j, aij in self.A[i]:
                w = aij * h
                zi = [a + w * b for a, b in zip(zi, k[j])]
            k.append(self.f(zi))
        return k

    def step(self, z, h):
        k = self._stages(z, h)
        out = list(z)
        for i, bi in self.B:
            w = bi * h
            out = [a + w * b for a, b in zip(out, k[i])]
        return self.arith.post_step(out)

    def step_err(self, z, h, scale):
        """Step and Hairer's scaled error norm in float64; `scale` = atol + rtol*|z|."""
        k = self._stages(z, h)
        out = list(z)
        for i, bi in self.B:
            w = bi * h
            out = [a + w * b for a, b in zip(out, k[i])]
        out = self.arith.post_step(out)
        kn = self.f(out)
        f = self.arith.to_float
        kf = np.array([[f(x) for x in ki] for ki in k])
        knf = np.array([f(x) for x in kn])
        err5 = (self.E5 @ kf) / scale
        err3 = (self.E3 @ kf) / scale
        if self.E5_extra:
            err5 = err5 + self.E5_extra * knf / scale
        if self.E3_extra:
            err3 = err3 + self.E3_extra * knf / scale
        n5, n3 = (err5**2).sum(), (err3**2).sum()
        hf = abs(float(f(h)))
        if n5 == 0.0 and n3 == 0.0:
            return out, 0.0
        den = n5 + 0.01 * n3
        return out, hf * n5 / np.sqrt(den * len(z))
