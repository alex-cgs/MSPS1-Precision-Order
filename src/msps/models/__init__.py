"""Two-constant error model, crossover order, out-of-sample checks (spec §4.4)."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

LOG2 = math.log10(2.0)


def fit_local_law(ns, medians, floor=-110.0):
    """log10 eps = icpt + slope (n+1), medians above the reference's noise floor (as the pilot)."""
    ns, med = np.asarray(ns, float), np.asarray(medians, float)
    keep = med > floor
    slope, icpt = np.polyfit(ns[keep] + 1, med[keep], 1)
    return float(slope), float(icpt)


def model_logE(ct, cr, slope, icpt, n, bits):
    """log10 of [10^ct eps_trunc(n) + 10^cr 2^-p], the amplification-free part of spec §4.4."""
    return np.logaddexp(np.log(10) * (ct + icpt + slope * (np.asarray(n, float) + 1)),
                        np.log(10) * (cr - np.asarray(bits, float) * LOG2)) / np.log(10)


def fit_constants(obs, slope, icpt, start=(7.0, 7.0)):
    """obs rows (n, bits, log10 E - log10 A). Nelder–Mead on squared decades, as in the pilot."""
    obs = np.asarray(obs, float)

    def loss(cc):
        return np.mean((obs[:, 2] - model_logE(cc[0], cc[1], slope, icpt, obs[:, 0], obs[:, 1])) ** 2)

    fit = minimize(loss, list(start), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000})
    return {"c_t": float(fit.x[0]), "c_r": float(fit.x[1]), "rms": float(np.sqrt(fit.fun)), "n_obs": int(len(obs))}


def crossover(ct, cr, slope, icpt, bits):
    """n*(p): equality of the two terms (spec §4.4), with log10 K = icpt and log10(rho/h) = -slope."""
    return (cr - np.asarray(bits, float) * LOG2 - ct - icpt) / slope - 1


def predicted_tpred(t_grid, logA, ct, cr, slope, icpt, n, bits, tol):
    v = model_logE(ct, cr, slope, icpt, n, bits) + np.asarray(logA)
    hit = np.nonzero(v > math.log10(tol))[0]
    return (float(t_grid[hit[0]]), False) if hit.size else (float(t_grid[-1]), True)


def resolved_logE(logA_steps, log_inject_trunc, log_inject_round):
    """Time-resolved variant (spec §4.4 known weakness): E(t) = sum_k eps_k A(t)/A(t_k), per step k along the reference.

    logA_steps: log10 cumulative growth at each step; log_inject_*: log10 of the error injected at each step.
    Returns log10 E at each step with both channels combined, via a running log-sum.
    """
    a = np.asarray(logA_steps, float)
    inj = np.logaddexp(np.log(10) * np.asarray(log_inject_trunc, float), np.log(10) * np.asarray(log_inject_round, float))
    x = inj - np.log(10) * a
    return (np.logaddexp.accumulate(x) + np.log(10) * a) / np.log(10)


def empirical_crossover(ns, tp, frac=0.95):
    """Smallest n whose T_pred reaches frac of the plateau of this (p, ic); plateau = median of the top three orders."""
    ns, tp = np.asarray(ns, float), np.asarray(tp, float)
    order = np.argsort(ns)
    ns, tp = ns[order], tp[order]
    plateau = float(np.median(tp[-3:]))
    ok = np.nonzero(tp >= frac * plateau)[0]
    return float(ns[ok[0]]) if ok.size else float("nan")
