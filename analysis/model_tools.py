"""Observation tables for the error model of spec §4.4, pilot selection rules kept: t > T/12, 0 < E < 1e-2."""
from __future__ import annotations

import numpy as np
import pandas as pd
from common import load_data
from msps.models import fit_constants, fit_local_law, model_logE
from scipy.optimize import minimize


def local_fit(local: pd.DataFrame, case: str, ics=None) -> tuple:
    d = local[local["case"] == case]
    if ics is not None:
        d = d[d["ic"].isin(ics)]
    med = d.groupby("n")["log10_eps"].median()
    return fit_local_law(med.index.values, med.values)


def run_series(path, T, stride=1):
    d = load_data(path)
    t, E = d["t"].values, d["E"].values
    keep = (t > T / 12) & (E > 0) & (E < 1e-2) & np.isfinite(E)
    return t[keep][::stride], E[keep][::stride]


def obs_one_curve(runs: pd.DataFrame, amp: dict, T: float, stride=1) -> np.ndarray:
    """Rows (n, bits, log10 E - log10 A, ic, t)."""
    out = []
    for r in runs.itertuples():
        t, E = run_series(r.run, T, stride)
        if t.size == 0:
            continue
        a = amp[(r.case, r.ic)]
        logA = np.interp(t, a["t"].values, np.log10(a["A"].values))
        out.append(np.column_stack([np.full(t.size, r.n), np.full(t.size, r.bits), np.log10(E) - logA, np.full(t.size, r.ic), t]))
    return np.vstack(out) if out else np.empty((0, 5))


def resolved_terms(a: pd.DataFrame, ns) -> dict:
    """Per order n: log E_trunc/10^ct and log E_round/(10^cr 2^-p) along the amplification grid (natural logs).

    Spec §4.4 fix for the known weakness: each step injects K (h_k/rho_k)^(n+1) (truncation) or the rounding unit,
    carried to t by the renormalized growth G(t)/G(t_k). rho is the reference's own radius estimate. Log-sums keep range.
    """
    g = a["log_growth"].values
    h, rho = a["h"].values, a["rho"].values
    rho = np.where(np.isfinite(rho) & (rho > 0), rho, np.nanmax(rho[np.isfinite(rho)]))
    ratio = np.log(np.maximum(h, 1e-300)) - np.log(rho)
    logR = np.logaddexp.accumulate(-g) + g
    out = {"round": logR}
    for n in ns:
        out[n] = np.logaddexp.accumulate((n + 1) * ratio - g) + g
    return out


def obs_resolved(runs: pd.DataFrame, amp: dict, T: float, stride=1) -> np.ndarray:
    """Rows (log10 E, trunc term log10, round term log10 without 2^-p, bits, ic, n, t)."""
    out = []
    terms_cache = {}
    ns = sorted(runs["n"].unique())
    for r in runs.itertuples():
        t, E = run_series(r.run, T, stride)
        if t.size == 0:
            continue
        key = (r.case, r.ic)
        a = amp[key]
        if key not in terms_cache:
            terms_cache[key] = resolved_terms(a, ns)
        tt = a["t"].values
        Tn = np.interp(t, tt, terms_cache[key][r.n]) / np.log(10)
        Rn = np.interp(t, tt, terms_cache[key]["round"]) / np.log(10)
        out.append(np.column_stack([np.log10(E), Tn, Rn, np.full(t.size, r.bits), np.full(t.size, r.ic), np.full(t.size, r.n), t]))
    return np.vstack(out) if out else np.empty((0, 7))


def resolved_pred(c, ob):
    LOG2 = np.log10(2.0)
    return np.logaddexp(np.log(10) * (c[0] + ob[:, 1]), np.log(10) * (c[1] + ob[:, 2] - ob[:, 3] * LOG2)) / np.log(10)


def fit_resolved(ob, start=(0.0, 0.0)) -> dict:
    fit = minimize(lambda c: np.mean((ob[:, 0] - resolved_pred(c, ob)) ** 2), list(start), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000})
    return {"c_t": float(fit.x[0]), "c_r": float(fit.x[1]), "rms": float(np.sqrt(fit.fun)), "n_obs": int(len(ob))}


def one_curve_pred(fit, slope, icpt, ob):
    return model_logE(fit["c_t"], fit["c_r"], slope, icpt, ob[:, 0], ob[:, 1])


__all__ = ["local_fit", "run_series", "obs_one_curve", "obs_resolved", "fit_constants", "fit_resolved", "resolved_pred", "one_curve_pred", "resolved_terms"]
