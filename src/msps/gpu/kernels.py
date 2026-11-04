"""Batched Taylor kernels for ensembles: f32, f64, dd over numpy (CPU reference) or JAX (GPU), one algorithm (spec §11).

Differences from the scalar path in msps.integrators.taylor, on purpose:
- a Cauchy sum is a fixed binary-tree fold of a masked vector, so the summation order is defined and equal on all
  back-ends rather than left to XLA's reduce;
- TwoProd uses Dekker's split, because XLA exposes no explicit FMA.
Any GPU-side contraction of a*b+c into FMA would break bitwise equality with numpy, which is what E10 measures.
"""
from __future__ import annotations

import numpy as np

SPLIT = 134217729.0


class _Float:
    """Plain IEEE: an element is one array."""

    def __init__(self, xp, dtype):
        self.xp, self.dtype = xp, dtype
        self.parts = 1

    def constant(self, q):
        return self.xp.asarray(float(q), dtype=self.dtype)

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def neg(self, a):
        return -a

    def mul(self, a, b):
        return a * b

    def divk(self, a, k):
        return a / self.xp.asarray(k, dtype=self.dtype)

    def where(self, mask, a):
        return self.xp.where(mask, a, self.xp.zeros_like(a))

    def lead(self, a):
        return a


class _DD:
    """Double-double: an element is a pair (hi, lo) of arrays; stacked along a leading axis of length 2."""

    def __init__(self, xp, dtype):
        self.xp, self.dtype = xp, dtype
        self.parts = 2

    def constant(self, q):
        from fractions import Fraction
        hi = float(q)
        lo = float(Fraction(q) - Fraction(hi))
        return self.xp.asarray(hi, dtype=self.dtype), self.xp.asarray(lo, dtype=self.dtype)

    @staticmethod
    def _two_sum(a, b):
        s = a + b
        bb = s - a
        return s, (a - (s - bb)) + (b - bb)

    @staticmethod
    def _quick(a, b):
        s = a + b
        return s, b - (s - a)

    @staticmethod
    def _two_prod(a, b):
        p = a * b
        t = SPLIT * a
        ah = t - (t - a)
        al = a - ah
        t = SPLIT * b
        bh = t - (t - b)
        bl = b - bh
        return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl

    def add(self, a, b):
        s, e = self._two_sum(a[0], b[0])
        t, f = self._two_sum(a[1], b[1])
        e = e + t
        s, e = self._quick(s, e)
        e = e + f
        return self._quick(s, e)

    def neg(self, a):
        return -a[0], -a[1]

    def sub(self, a, b):
        return self.add(a, (-b[0], -b[1]))

    def mul(self, a, b):
        p, e = self._two_prod(a[0], b[0])
        e = e + a[0] * b[1] + a[1] * b[0]
        return self._quick(p, e)

    def divk(self, a, k):
        d = self.xp.asarray(k, dtype=self.dtype)
        q1 = a[0] / d
        p, e = self._two_prod(q1, d)
        r = self.add(a, (-p, -e))
        q2 = r[0] / d
        p, e = self._two_prod(q2, d)
        r = self.add(r, (-p, -e))
        q3 = r[0] / d
        s, e = self._quick(q1, q2)
        return self.add((s, e), (q3, self.xp.zeros_like(q3)))

    def where(self, mask, a):
        z = self.xp.zeros_like(a[0])
        return self.xp.where(mask, a[0], z), self.xp.where(mask, a[1], z)

    def lead(self, a):
        return a[0]


def _ops(prec, xp):
    if prec == "f32":
        return _Float(xp, xp.float32)
    if prec == "f64":
        return _Float(xp, xp.float64)
    if prec == "dd":
        return _DD(xp, xp.float64)
    raise ValueError(prec)


class BatchedTaylor:
    """step(z, h), advance(z, t, target): z is (D, batch) for f32/f64 and a pair of those for dd; t is float64."""

    def __init__(self, tape, n: int, prec: str, backend: str = "numpy", safety: float = 8.0, cap: float = 0.5, device=None):
        self.tape, self.n, self.prec, self.backend = tape, int(n), prec, backend
        self.safety, self.cap = float(safety), float(cap)
        self.device = device
        if backend == "numpy":
            import numpy as xp
            self.jax = None
        else:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as xp
            self.jax = jax
        self.xp = xp
        self.ops = _ops(prec, xp)
        self.K = [self.ops.constant(c) for c in tape.constants]
        self._compile()

    def _empty(self, batch):
        xp = self.xp
        shape = (self.tape.n_nodes, self.n + 1, batch)
        dt = self.ops.dtype
        return xp.zeros(shape, dtype=dt) if self.ops.parts == 1 else (xp.zeros(shape, dtype=dt), xp.zeros(shape, dtype=dt))

    def _set(self, S, i, k, v):
        if self.backend == "numpy":
            if self.ops.parts == 1:
                S[i, k] = v
            else:
                S[0][i, k] = v[0]
                S[1][i, k] = v[1]
            return S
        if self.ops.parts == 1:
            return S.at[i, k].set(v)
        return S[0].at[i, k].set(v[0]), S[1].at[i, k].set(v[1])

    def _get(self, S, i, k=None):
        if self.ops.parts == 1:
            return S[i] if k is None else S[i, k]
        return (S[0][i], S[1][i]) if k is None else (S[0][i, k], S[1][i, k])

    def _tree_sum(self, v):
        """Fold over axis 0 with a fixed pairing; depends only on the vector length n+1."""
        xp, ops = self.xp, self.ops
        m = self.n + 1
        while m > 1:
            half = m // 2
            if ops.parts == 1:
                paired = ops.add(v[0:2 * half:2], v[1:2 * half:2])
                v = xp.concatenate([paired, v[2 * half:m]], axis=0) if m % 2 else paired
            else:
                a, b = (v[0][0:2 * half:2], v[1][0:2 * half:2]), (v[0][1:2 * half:2], v[1][1:2 * half:2])
                p = ops.add(a, b)
                v = (xp.concatenate([p[0], v[0][2 * half:m]], axis=0), xp.concatenate([p[1], v[1][2 * half:m]], axis=0)) if m % 2 else p
            m = half + (m % 2)
        return v[0] if ops.parts == 1 else (v[0][0], v[1][0])

    def _order_body(self, S, k):
        """All tape nodes at order k, then state coefficients at order k+1 (spec §4.2)."""
        xp, ops = self.xp, self.ops
        ls = xp.arange(self.n + 1)
        mask = (ls <= k)[:, None]
        rev = xp.clip(k - ls, 0, self.n)
        for kind, dst, a, b in self.tape.ops:
            if kind == "const":
                continue
            if kind == "add":
                v = ops.add(self._get(S, a, k), self._get(S, b, k))
            elif kind == "sub":
                v = ops.sub(self._get(S, a, k), self._get(S, b, k))
            elif kind == "neg":
                v = ops.neg(self._get(S, a, k))
            elif kind == "scale":
                v = ops.mul(self._get(S, a, k), self.K[b])
            elif kind == "mul":
                A = self._get(S, a)
                B = self._get(S, b)
                if ops.parts == 1:
                    p = ops.mul(A, B[rev])
                else:
                    p = ops.mul(A, (B[0][rev], B[1][rev]))
                v = self._tree_sum(ops.where(mask, p))
            else:
                raise ValueError(kind)
            S = self._set(S, dst, k, v)
        for i in range(self.tape.dim):
            S = self._set(S, i, k + 1, ops.divk(self._get(S, self.tape.rhs[i], k), k + 1))
        return S

    def _coeffs(self, z):
        """Node series with state coefficients 0..n, for one batch."""
        xp, ops = self.xp, self.ops
        batch = z.shape[1] if ops.parts == 1 else z[0].shape[1]
        S = self._empty(batch)
        for i in range(self.tape.dim):
            S = self._set(S, i, 0, z[i] if ops.parts == 1 else (z[0][i], z[1][i]))
        for kind, dst, a, b in self.tape.ops:
            if kind == "const":
                c = self.K[a]
                val = xp.broadcast_to(c, (batch,)) if ops.parts == 1 else (xp.broadcast_to(c[0], (batch,)), xp.broadcast_to(c[1], (batch,)))
                S = self._set(S, dst, 0, val)
        if self.jax is not None:
            S, _ = self.jax.lax.scan(lambda s, k: (self._order_body(s, k), None), S, xp.arange(self.n))
        else:
            for k in range(self.n):
                S = self._order_body(S, k)
        return S

    def _horner(self, S, h):
        """h: batch float array in the kernel's working type."""
        ops, xp = self.ops, self.xp
        D = self.tape.dim
        if ops.parts == 1:
            acc = S[:D, self.n]
            for j in range(self.n - 1, -1, -1):
                acc = acc * h + S[:D, j]
            return acc
        hh = (h, xp.zeros_like(h))
        acc = (S[0][:D, self.n], S[1][:D, self.n])
        for j in range(self.n - 1, -1, -1):
            acc = ops.add(ops.mul(acc, hh), (S[0][:D, j], S[1][:D, j]))
        return acc

    def _radius(self, S):
        xp = self.xp
        lead = S if self.ops.parts == 1 else S[0]
        D = self.tape.dim
        e = []
        for k in (self.n - 1, self.n):
            mk = xp.max(xp.abs(lead[:D, k]), axis=0).astype(xp.float64)
            e.append(xp.where(mk > 0, mk ** (-1.0 / k), xp.inf))
        return xp.minimum(e[0], e[1])

    def _adaptive_h(self, S, t, target):
        """h = 2^floor(log2(rho/safety)), capped, landing exactly on `target` (dyadic in float64, spec §7.1)."""
        xp = self.xp
        rho = self._radius(S)
        ex = xp.floor(xp.log2(rho / self.safety))
        ex = xp.where(xp.isfinite(ex), ex, 0.0)
        hmax = xp.minimum(xp.ldexp(xp.ones_like(ex), ex.astype(xp.int32)), self.cap)
        return xp.minimum(hmax, target - t)

    def _step_fn(self, z, h):
        S = self._coeffs(z)
        return self._horner(S, h.astype(self.ops.dtype))

    def _adv_body(self, z, t, target, steps):
        S = self._coeffs(z)
        h = self._adaptive_h(S, t, target)
        new_t = t + h
        znew = self._horner(S, h.astype(self.ops.dtype))
        return znew, new_t, steps + (h > 0)

    def _compile(self):
        if self.jax is None:
            self.step = self._step_fn
            self._adv = None
            return
        jax, xp = self.jax, self.xp
        self.step = jax.jit(self._step_fn, device=self.device) if self.device is not None else jax.jit(self._step_fn)

        def adv(z, t, target, steps):
            def cond(state):
                return xp.any(state[1] < target)

            def body(state):
                zz, tt, ss = state
                return self._adv_body(zz, tt, target, ss)

            return jax.lax.while_loop(cond, body, (z, t, steps))

        self._adv = jax.jit(adv, device=self.device) if self.device is not None else jax.jit(adv)

    def advance(self, z, t, target, max_iter: int = 2_000_000):
        """Per-member adaptive steps until every member sits on `target` (float64 scalar). Returns (z, t, steps)."""
        xp = self.xp
        steps = xp.zeros_like(t).astype(xp.int64)
        if self._adv is not None:
            return self._adv(z, t, xp.asarray(target, xp.float64), steps)
        for _ in range(max_iter):
            if not bool((t < target).any()):
                break
            z, t, steps = self._adv_body(z, t, target, steps)
        return z, t, steps

    def from_fractions(self, states: list):
        """states: list of exact rational state vectors. Same conversion on every back-end."""
        from fractions import Fraction
        arr = np.array([[float(q) for q in st] for st in states], dtype=np.float64).T
        dt = np.float32 if self.prec == "f32" else np.float64
        if self.prec == "dd":
            lo = np.array([[float(Fraction(q) - Fraction(float(q))) for q in st] for st in states], dtype=np.float64).T
            return self.xp.asarray(arr), self.xp.asarray(lo)
        return self.xp.asarray(arr.astype(dt))

    def to_numpy(self, z):
        if self.ops.parts == 1:
            return np.asarray(z)
        return np.asarray(z[0]), np.asarray(z[1])


def compare_states(a, b) -> dict:
    """Bitwise and magnitude comparison of two batched states of one kernel type."""
    if isinstance(a, tuple):
        same = (a[0] == b[0]) & (a[1] == b[1])
        mag = np.abs((a[0] - b[0]) + (a[1] - b[1]))
    else:
        same = a == b
        mag = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"bitwise_identical_fraction": float(same.mean()), "max_abs_diff": float(np.nanmax(mag)), "members_identical": float(same.all(axis=0).mean())}
