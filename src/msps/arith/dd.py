"""Double-double numbers from error-free transformations (Hida, Li, Bailey QD algorithms; about 106 bits)."""
from __future__ import annotations

import math

_fma = getattr(math, "fma", None)
_SPLIT = 134217729.0  # 2^27 + 1


if _fma is not None:
    def _two_prod(a, b):
        p = a * b
        return p, _fma(a, b, -p)
else:
    def _two_prod(a, b):
        p = a * b
        t = _SPLIT * a; ah = t - (t - a); al = a - ah
        t = _SPLIT * b; bh = t - (t - b); bl = b - bh
        return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


class DD:
    __slots__ = ("hi", "lo")

    def __init__(self, hi=0.0, lo=0.0):
        self.hi = hi
        self.lo = lo

    def __add__(self, o):
        if not isinstance(o, DD):
            o = DD(float(o))
        a, b = self.hi, o.hi
        s = a + b
        bb = s - a
        e = (a - (s - bb)) + (b - bb)
        c, d = self.lo, o.lo
        t = c + d
        bb = t - c
        f = (c - (t - bb)) + (d - bb)
        e += t
        r = s + e
        e -= r - s
        e += f
        hi = r + e
        return DD(hi, e - (hi - r))

    __radd__ = __add__

    def __neg__(self):
        return DD(-self.hi, -self.lo)

    def __sub__(self, o):
        if not isinstance(o, DD):
            o = DD(float(o))
        return self + DD(-o.hi, -o.lo)

    def __rsub__(self, o):
        return DD(float(o)) + DD(-self.hi, -self.lo)

    def __mul__(self, o):
        if isinstance(o, DD):
            p, e = _two_prod(self.hi, o.hi)
            e += self.hi * o.lo + self.lo * o.hi
        else:
            o = float(o)
            p, e = _two_prod(self.hi, o)
            e += self.lo * o
        hi = p + e
        return DD(hi, e - (hi - p))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, (int, float)):
            o = DD(float(o))
        q1 = self.hi / o.hi
        r = self - o * q1
        q2 = r.hi / o.hi
        r = r - o * q2
        q3 = r.hi / o.hi
        s = q1 + q2
        lo = q2 - (s - q1)
        return DD(s, lo) + DD(q3)

    def __rtruediv__(self, o):
        return DD(float(o)) / self

    def __abs__(self):
        return -self if self.hi < 0.0 or (self.hi == 0.0 and self.lo < 0.0) else self

    def __lt__(self, o):
        if not isinstance(o, DD):
            o = DD(float(o))
        return self.hi < o.hi or (self.hi == o.hi and self.lo < o.lo)

    def __gt__(self, o):
        if not isinstance(o, DD):
            o = DD(float(o))
        return self.hi > o.hi or (self.hi == o.hi and self.lo > o.lo)

    def __float__(self):
        return self.hi

    def sqrt(self):
        if self.hi <= 0.0:
            return DD(math.sqrt(self.hi) if self.hi == 0.0 else float("nan"))
        x = 1.0 / math.sqrt(self.hi)
        ax = self.hi * x
        d = self - DD(ax) * ax
        return DD(ax) + DD(d.hi * (x * 0.5))

    def __repr__(self):
        return f"DD({self.hi!r}, {self.lo!r})"

    def __getstate__(self):
        return (self.hi, self.lo)

    def __setstate__(self, s):
        self.hi, self.lo = s
