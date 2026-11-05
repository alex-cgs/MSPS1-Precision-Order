"""Gauss–Legendre IRK with s stages, order 2s; fixed-point stage iteration with a recorded tolerance (spec §7.1).

The tolerance is a second precision knob (RQ7): `tol` bounds max|delta K|*h between sweeps; `iterations` records the
total sweeps so cost can be charged per RHS evaluation. On lifted systems GL keeps quadratic invariants; it is
canonical-symplectic only on the original (q, v) variables.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb

import mpmath as mp

from .rk import _RHS


def tableau(s: int, digits: int = 320):
    """(A, b) as exact Fractions approximating the GL coefficients to `digits` digits."""
    with mp.workdps(digits):
        # nodes: roots of the shifted Legendre polynomial, from its exact coefficients, highest degree first
        coeffs = [mp.mpf((-1) ** (s + j) * comb(s, j) * comb(s + j, j)) for j in range(s, -1, -1)]
        c = sorted(mp.re(r) for r in mp.polyroots(coeffs, maxsteps=500, extraprec=2 * digits))
        V = mp.matrix(s, s)
        for i in range(s):
            for j in range(s):
                V[i, j] = c[j] ** i
        b = mp.lu_solve(V, mp.matrix([mp.mpf(1) / (i + 1) for i in range(s)]))
        A = []
        for r in range(s):
            row = mp.lu_solve(V, mp.matrix([c[r] ** (i + 1) / (i + 1) for i in range(s)]))
            A.append([row[j] for j in range(s)])
        to_q = lambda x: Fraction(int(mp.floor(x * mp.mpf(10) ** digits)), 10**digits)  # noqa: E731
        return [[to_q(x) for x in row] for row in A], [to_q(b[i]) for i in range(s)]


class GaussLegendre:
    name = "gl"

    def __init__(self, case, arith, s: int, tol: float, max_iter: int = 200):
        self.f = _RHS(case, arith)
        self.arith = arith
        self.s = int(s)
        self.order = 2 * self.s
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        A, b = tableau(self.s)
        self.A = [[arith.convert(x) for x in row] for row in A]
        self.b = [arith.convert(x) for x in b]
        self.iterations = 0
        self.not_converged = 0
        self._last_k = None

    @property
    def evals_per_step(self):
        return self.s

    def step(self, z, h):
        f, A, s = self.f, self.A, self.s
        to_f = self.arith.to_float
        hf = abs(float(to_f(h)))
        k0 = f(z)
        K = [list(k0) for _ in range(s)]
        for it in range(self.max_iter):
            newK = []
            for i in range(s):
                zi = list(z)
                for j in range(s):
                    w = A[i][j] * h
                    zi = [a + w * b for a, b in zip(zi, K[j])]
                newK.append(f(zi))
            delta = 0.0
            for i in range(s):
                for a, b in zip(K[i], newK[i]):
                    delta = max(delta, abs(to_f(a - b)))
            K = newK
            self.iterations += 1
            if delta * hf <= self.tol:
                break
        else:
            self.not_converged += 1
        out = list(z)
        for i in range(s):
            w = self.b[i] * h
            out = [a + w * b for a, b in zip(out, K[i])]
        return self.arith.post_step(out)
