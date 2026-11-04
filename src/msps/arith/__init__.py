"""Arithmetic back-ends behind one interface (spec §7.2).

Every back-end's numbers support +, -, *, and / by int, so the tape-generated code (msps.symbolic.tape)
is the single code path of spec §1. `math.fma` needs Python >= 3.13; dd.py falls back to Dekker's split.
"""
from __future__ import annotations

import math
from fractions import Fraction

import gmpy2
import numpy as np

from .dd import DD

# working precision for error measurement against references, never for the run's own arithmetic
DIAG_BITS = 2048


def _frac_to_mpfr(q: Fraction):
    return gmpy2.mpfr(gmpy2.mpq(q.numerator, q.denominator))


class Arithmetic:
    name: str = ""
    bits: int | None = None
    exact: bool = False

    def activate(self):
        """Set per-process state. Must run before any arithmetic in this process (spec §7.2, determinism)."""
        gmpy2.get_context().precision = DIAG_BITS

    def convert(self, q: Fraction):
        raise NotImplementedError

    def to_float(self, x) -> float:
        return float(x)

    def to_fraction(self, x) -> Fraction:
        raise NotImplementedError

    def to_mpfr(self, x):
        """Exact (or context-rounded) value as mpfr, for diagnostics."""
        return _frac_to_mpfr(self.to_fraction(x))

    def unit_roundoff(self) -> float | None:
        return None if self.bits is None else 2.0 ** (-self.bits)

    def post_step(self, z):
        return z

    def sqrt(self, x):
        raise NotImplementedError(f"{self.name} has no sqrt")

    def sin(self, x):
        raise NotImplementedError(f"{self.name} has no sin")

    def cos(self, x):
        raise NotImplementedError(f"{self.name} has no cos")

    def diff_abs(self, x, ref) -> float:
        """|x - ref| as float64 with ref an mpfr at reference precision; x - ref must not round to noise."""
        return abs(float(self.to_mpfr(x) - ref))

    def isfinite(self, x) -> bool:
        return math.isfinite(self.to_float(x))


class _IEEE(Arithmetic):
    dtype = None

    def __init__(self):
        self._t = self.dtype

    def convert(self, q):
        return self._t(_ieee_round(q, self.bits))

    def to_float(self, x):
        return float(x)

    def to_fraction(self, x):
        return Fraction(float(x))

    def sqrt(self, x):
        return np.sqrt(x)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def diff_abs(self, x, ref):
        return abs(float(gmpy2.mpfr(float(x)) - ref))


def _ieee_round(q: Fraction, bits: int) -> float:
    """Correctly rounded conversion (avoids double rounding through float64 for bits < 53)."""
    if bits >= 53:
        return q.numerator / q.denominator if abs(q.numerator) < 2**1000 and q.denominator < 2**1000 else float(q)
    with gmpy2.context(gmpy2.get_context(), precision=bits):
        return float(_frac_to_mpfr(q))


class F16(_IEEE):
    name, bits, dtype = "f16", 11, np.float16


class F32(_IEEE):
    name, bits, dtype = "f32", 24, np.float32


class F64(Arithmetic):
    name, bits = "f64", 53

    def convert(self, q):
        return float(q)

    def to_fraction(self, x):
        return Fraction(x)

    def sqrt(self, x):
        return math.sqrt(x)

    def sin(self, x):
        return math.sin(x)

    def cos(self, x):
        return math.cos(x)

    def diff_abs(self, x, ref):
        return abs(float(gmpy2.mpfr(x) - ref))


class F80(Arithmetic):
    """x87 extended via numpy.longdouble; 64-bit significand on this platform (checked in tests)."""
    name, bits = "f80", 64

    def convert(self, q):
        L = np.longdouble
        return L(q.numerator) / L(q.denominator) if abs(q.numerator) < 2**60 and q.denominator < 2**60 else self._from_big(q)

    @staticmethod
    def _from_big(q):
        with gmpy2.context(gmpy2.get_context(), precision=64):
            x = _frac_to_mpfr(q)
        m, e = x.as_mantissa_exp()
        return np.ldexp(np.longdouble(int(m)), int(e))

    def to_float(self, x):
        return float(x)

    def to_fraction(self, x):
        m, e = np.frexp(x)
        return Fraction(int(np.ldexp(m, 64))) * Fraction(2) ** (int(e) - 64)

    def sqrt(self, x):
        return np.sqrt(x)


class DoubleDouble(Arithmetic):
    name, bits = "dd", 106

    def convert(self, q):
        hi = float(q)
        lo = float(q - Fraction(hi))
        return DD(*_fast2sum(hi, lo))

    def to_float(self, x):
        return x.hi

    def to_fraction(self, x):
        return Fraction(x.hi) + Fraction(x.lo)

    def to_mpfr(self, x):
        return gmpy2.mpfr(x.hi) + gmpy2.mpfr(x.lo)

    def isfinite(self, x):
        return math.isfinite(x.hi) and math.isfinite(x.lo)

    def sqrt(self, x):
        return x.sqrt()

    def diff_abs(self, x, ref):
        return abs(float(gmpy2.mpfr(x.hi) + gmpy2.mpfr(x.lo) - ref))


def _fast2sum(a, b):
    s = a + b
    return s, b - (s - a)


class MPFR(Arithmetic):
    def __init__(self, bits: int):
        self.bits = int(bits)
        self.name = f"mpfr-{self.bits}"

    def activate(self):
        gmpy2.get_context().precision = self.bits

    def convert(self, q):
        return _frac_to_mpfr(q)

    def to_float(self, x):
        return float(x)

    def to_fraction(self, x):
        return Fraction(*map(int, gmpy2.mpq(x).as_integer_ratio()))

    def to_mpfr(self, x):
        return x

    def sqrt(self, x):
        return gmpy2.sqrt(x)

    def sin(self, x):
        return gmpy2.sin(x)

    def cos(self, x):
        return gmpy2.cos(x)

    def diff_abs(self, x, ref):
        with gmpy2.context(gmpy2.get_context(), precision=DIAG_BITS):
            return abs(float(x - ref))

    def isfinite(self, x):
        return gmpy2.is_finite(x)


class Rational(Arithmetic):
    name, bits, exact = "q", None, True

    def __init__(self):
        import flint
        self._fmpq = flint.fmpq

    def convert(self, q):
        return self._fmpq(q.numerator, q.denominator)

    def to_float(self, x):
        return _fmpq_to_float(x)

    def to_fraction(self, x):
        return Fraction(int(x.p), int(x.q))

    def isfinite(self, x):
        return True


def _fmpq_to_float(x) -> float:
    with gmpy2.context(gmpy2.get_context(), precision=64):
        return float(gmpy2.mpq(int(x.p), int(x.q)))


class RationalRounded(Rational):
    """qr-p: exact fmpq within a step; every state component rounded to the nearest multiple of 2^-p after it.

    Fixed-point rounding, so it differs from float's relative rounding (the point of E9, RQ6).
    """
    exact = False

    def __init__(self, bits: int):
        super().__init__()
        self.bits = int(bits)
        self.name = f"qr-{self.bits}"
        self._scale = 1 << self.bits

    def unit_roundoff(self):
        return 2.0 ** (-self.bits - 1)

    def _round(self, x):
        p, q = int(x.p), int(x.q)
        return self._fmpq((2 * p * self._scale + q) // (2 * q), self._scale)

    def post_step(self, z):
        return [self._round(x) for x in z]

    def convert(self, q):
        return self._fmpq(q.numerator, q.denominator)


class Arb(Arithmetic):
    """Ball arithmetic: a run's radius is a certified bound on its accumulated rounding and truncation
    only when the truncation remainder is enclosed, which this project does not do; radius = rounding only."""

    def __init__(self, bits: int):
        import flint
        self._flint = flint
        self.bits = int(bits)
        self.name = f"arb-{self.bits}"

    def activate(self):
        super().activate()
        self._flint.ctx.prec = self.bits

    def convert(self, q):
        return self._flint.arb(self._flint.fmpq(q.numerator, q.denominator))

    def to_float(self, x):
        return float(x.mid())

    def to_fraction(self, x):
        man, exp = x.mid().man_exp()
        return Fraction(int(man)) * Fraction(2) ** int(exp)

    def radius(self, x) -> float:
        return float(x.rad())

    def sqrt(self, x):
        return x.sqrt()


def get_arith(spec: str) -> Arithmetic:
    """Back-end from its spec §7.2 ID: f16 f32 f64 f80 dd mpfr-<p> arb-<p> q qr-<p>."""
    simple = {"f16": F16, "f32": F32, "f64": F64, "f80": F80, "dd": DoubleDouble, "q": Rational}
    if spec in simple:
        return simple[spec]()
    kind, _, b = spec.partition("-")
    if kind == "mpfr":
        return MPFR(int(b))
    if kind == "qr":
        return RationalRounded(int(b))
    if kind == "arb":
        return Arb(int(b))
    raise ValueError(f"unknown arithmetic {spec!r}")
