"""Taylor method of order n; recurrences from the generated tape, Horner evaluation (spec §4.2, §7.1)."""
from __future__ import annotations

from fractions import Fraction


class TaylorStepper:
    name = "taylor"

    def __init__(self, case, n: int, arith):
        self.case, self.n, self.arith = case, int(n), arith
        self._f = case.coeffs_fn
        self.K = [arith.convert(c) for c in case.tape.constants]
        self.ZERO = arith.convert(Fraction(0))

    @property
    def order(self):
        return self.n

    def coeffs(self, z):
        return self._f(z, self.n, self.K, self.ZERO)

    def horner(self, c, h):
        out = []
        for ci in c:
            acc = ci[-1]
            for a in ci[-2::-1]:
                acc = acc * h + a
            out.append(acc)
        return out

    def step(self, z, h):
        return self.arith.post_step(self.horner(self.coeffs(z), h))

    def radius(self, c) -> float:
        """min over the last two orders of (max_i |x_[k]|)^(-1/k), as in the pilot."""
        f = self.arith.to_float
        n = len(c[0]) - 1
        best = float("inf")
        for k in (n - 1, n):
            mk = max(abs(f(v[k])) for v in c)
            if mk > 0 and mk != float("inf"):
                best = min(best, mk ** (-1.0 / k))
            elif mk == float("inf"):
                best = 0.0
        return best
