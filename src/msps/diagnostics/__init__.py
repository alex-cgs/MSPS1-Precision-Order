"""Metrics of spec §4.7. Definitions fixed by the author; do not change without him."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy import stats as sstats


def error_positions(arith, z, ref, idx) -> float:
    """E(t): max over position components of |q_i - q_i_ref| (spec §4.7). inf for non-finite states."""
    best = 0.0
    for i in idx:
        try:
            e = arith.diff_abs(z[i], ref[i])
        except (OverflowError, ValueError, ZeroDivisionError):
            return math.inf
        if e != e:
            return math.inf
        if e > best:
            best = e
    return best


def tpred(t, E, tol: float, T: float) -> dict:
    """First grid time with E > tol; censored at T if never crossed before the run ended at T (spec §4.7)."""
    t, E = np.asarray(t, float), np.asarray(E, float)
    hit = np.nonzero(E > tol)[0]
    if hit.size:
        return {"tpred": float(t[hit[0]]), "censored": False}
    if t[-1] >= T - 1e-9:
        return {"tpred": float(T), "censored": True}
    return {"tpred": float(t[-1]), "censored": False}


class TpredTracker:
    """Online T_pred for several tolerances, so sweeps need not store per-step E."""

    def __init__(self, tols, T):
        self.tols = [float(x) for x in tols]
        self.T = float(T)
        self.hit = {tol: None for tol in self.tols}
        self.last_t = 0.0

    def update(self, t: float, E: float):
        self.last_t = t
        for tol in self.tols:
            if self.hit[tol] is None and E > tol:
                self.hit[tol] = t

    def result(self) -> dict:
        out = {}
        for tol in self.tols:
            key = f"{tol:g}"
            if self.hit[tol] is not None:
                out[key] = {"tpred": self.hit[tol], "censored": False}
            elif self.last_t >= self.T - 1e-9:
                out[key] = {"tpred": self.T, "censored": True}
            else:
                out[key] = {"tpred": self.last_t, "censored": False, "stopped_early": True}
        return out


class InvariantMonitor:
    """Delta H(t) per conserved quantity, Delta C(t) = max|constraint| (spec §4.7)."""

    def __init__(self, case, arith, z0):
        self.arith = arith
        self.names = list(case.system.invariants)
        self.cnames = list(case.system.constraints)
        self.fi, ki = case.value_fn("invariants")
        self.fc, kc = case.value_fn("constraints")
        self.Ki = [arith.convert(x) for x in ki]
        self.Kc = [arith.convert(x) for x in kc]
        self.H0 = self.fi(z0, self.Ki) if self.names else []

    def deltas(self, z) -> dict:
        f = self.arith.to_float
        out = {}
        try:
            vals = self.fi(z, self.Ki) if self.names else []
            for name, v, v0 in zip(self.names, vals, self.H0):
                out[f"d{name}"] = abs(f(v - v0))
            if self.cnames:
                out["dC"] = max(abs(f(c)) for c in self.fc(z, self.Kc))
        except (OverflowError, ValueError, ZeroDivisionError):
            out = {f"d{n}": math.inf for n in self.names}
            if self.cnames:
                out["dC"] = math.inf
        return out

    def columns(self) -> list:
        return [f"d{n}" for n in self.names] + (["dC"] if self.cnames else [])


def pair_distances(S, z_float) -> list:
    N, dim = S.meta["N"], S.meta["dim"]
    return [math.sqrt(sum((z_float[a * N + i] - z_float[a * N + j]) ** 2 for a in range(dim))) for i, j in S.meta["pairs"]]


def amplification(sep, perturbation: float, floor_factor: float = 1e-3):
    """A(t) = separation of a perturbed reference divided by the perturbation size (spec §4.7); floored like the pilot."""
    return np.maximum(np.asarray(sep, float), perturbation * floor_factor) / perturbation


def ftle_from_log_growth(t, log_growth):
    """lambda_tau per renormalization interval from cumulative ln-growth values at times t (spec §4.5)."""
    t, g = np.asarray(t, float), np.asarray(log_growth, float)
    return (g[1:] - g[:-1]) / (t[1:] - t[:-1])


def wasserstein1_hist(p, q, edges) -> float:
    """W1 between two histograms on common bins, as integral |CDF_p - CDF_q|."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(np.abs(np.cumsum(p - q))[:-1] * np.diff(edges)[:-1]))


def ks_escape(a, b) -> dict:
    """Two-sample KS on escape times; censored runs dropped and their counts reported."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"ks": float("nan"), "p": float("nan"), "na": int(a.size), "nb": int(b.size)}
    r = sstats.ks_2samp(a, b)
    return {"ks": float(r.statistic), "p": float(r.pvalue), "na": int(a.size), "nb": int(b.size)}


def escape_state(S, zf, r_factor: float = 10.0):
    """Index of an escaping body or None.

    Project definition, since Stone & Leigh 2019 give no operational test (PROGRESS.md): body i is escaping when it is
    unbound from the barycentre of the other two (positive two-body energy, moving away) and its distance exceeds
    r_factor times the separation of the other two. Planar triples only.
    """
    masses = [float(m) for m in S.meta["masses"]]
    N = S.meta["N"]
    if N != 3:
        raise ValueError("escape_state is for three bodies")
    q = np.array(zf[:6]).reshape(2, 3)
    v = np.array(zf[6:12]).reshape(2, 3)
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        mjk = masses[j] + masses[k]
        cq = (masses[j] * q[:, j] + masses[k] * q[:, k]) / mjk
        cv = (masses[j] * v[:, j] + masses[k] * v[:, k]) / mjk
        dq, dv = q[:, i] - cq, v[:, i] - cv
        r = float(np.hypot(*dq))
        sep = float(np.hypot(*(q[:, j] - q[:, k])))
        mu_e = 0.5 * float(dv @ dv) - (masses[i] + mjk) / r
        if mu_e > 0 and float(dq @ dv) > 0 and r > r_factor * sep:
            return i
    return None



def censoring_fraction(flags) -> float:
    f = np.asarray(flags, bool)
    return float(f.mean()) if f.size else float("nan")


