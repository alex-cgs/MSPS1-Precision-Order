"""E2: T_pred tables with censoring, crossover n*(p) empirical and per-IC model-based, as median and IQR over ICs."""
import numpy as np
import pandas as pd
from common import amp_table, iqr, local_table, save_table, traj_rows
from model_tools import local_fit, obs_one_curve
from msps.models import crossover, empirical_crossover, fit_constants, predicted_tpred
from msps.systems import load_case

STUDY = "E2"
tr = traj_rows(STUDY)
sh = tr[tr.schedule == "shared"]
save_table(sh.drop(columns=["run", "method", "substeps", "s", "k", "rtol", "solver_tol", "solver_iterations", "solver_failures", "sample_all", "schedule"]), "e2_tpred_runs", md_rows=40)
summ = sh.groupby(["case", "arith", "bits", "n", "tol"]).agg(tpred_median=("tpred", "median"), tpred_q25=("tpred", lambda x: np.percentile(x, 25)),
                                                            tpred_q75=("tpred", lambda x: np.percentile(x, 75)), censored_frac=("censored", "mean"), runs=("tpred", "size"),
                                                            wall_median=("wall", "median")).reset_index()
save_table(summ, "e2_tpred_summary", md_rows=60)
ad = tr[tr.schedule == "adaptive"]
if len(ad):
    s2 = ad.groupby(["case", "arith", "bits", "n", "tol"]).agg(tpred_median=("tpred", "median"), censored_frac=("censored", "mean"), runs=("tpred", "size"), wall_median=("wall", "median"), steps_median=("steps", "median")).reset_index()
    save_table(s2, "e2_tpred_adaptive_summary", md_rows=40)
amp = amp_table(STUDY)
loc = local_table(STUDY)
emp_rows, mod_rows, fit_rows, pred_rows = [], [], [], []
for case, dc in sh.groupby("case"):
    T = float(load_case(case).card["T"])
    for tol, dt in dc.groupby("tol"):
        for (ic, arith), d in dt.groupby(["ic", "arith"]):
            d = d.sort_values("n")
            ns, tp = d.n.values, d.tpred.values
            at_max = tp[-1]
            plateau = (not d.censored.values[-1]) and abs(tp[-1] - tp[-3]) <= 0.05 * tp[-1] if len(tp) >= 3 else False
            emp_rows.append({"case": case, "tol": tol, "ic": ic, "arith": arith, "bits": d.bits.iloc[0],
                             "n_star_empirical": empirical_crossover(ns, tp) if plateau else np.nan, "plateau_reached": bool(plateau),
                             "plateau_tpred": at_max, "censored_at_max_n": bool(d.censored.values[-1])})
    d6 = dc[dc.tol == 1e-6]
    for ic, di in d6.groupby("ic"):
        if (case, ic) not in amp or ic not in loc[loc.case == case].ic.values:
            continue
        slope, icpt = local_fit(loc, case, [ic])
        ob = obs_one_curve(di, amp, T)
        if len(ob) < 20:
            fit_rows.append({"case": case, "ic": ic, "slope": slope, "log10_K": icpt, "n_obs": len(ob)})
            continue
        fit = fit_constants(ob[:, :3], slope, icpt)
        fit_rows.append({"case": case, "ic": ic, "slope": slope, "log10_K": icpt, **fit})
        for bits in sorted(di.bits.unique()):
            mod_rows.append({"case": case, "ic": ic, "bits": bits, "n_star_model": float(crossover(fit["c_t"], fit["c_r"], slope, icpt, bits))})
            a = amp[(case, ic)]
            for n in sorted(di.n.unique()):
                for tol in (1e-6, 1e-3):
                    tp, cens = predicted_tpred(a["t"].values, np.log10(a["A"].values), fit["c_t"], fit["c_r"], slope, icpt, n, bits, tol)
                    pred_rows.append({"case": case, "ic": ic, "n": n, "bits": bits, "tol": tol, "tpred_model": tp, "model_censored": cens})
emp = pd.DataFrame(emp_rows)
save_table(emp, "e2_crossover_runs_empirical", md_rows=30)
mod = pd.DataFrame(mod_rows)
save_table(mod, "e2_crossover_runs_model", md_rows=30)
save_table(pd.DataFrame(fit_rows), "e2_model_fits_per_ic", md_rows=30)
save_table(pd.DataFrame(pred_rows), "e2_model_tpred", md_rows=30)
rows = []
for (case, bits, tol), d in emp.groupby(["case", "bits", "tol"]):
    v = d.n_star_empirical.dropna()
    q = iqr(v) if len(v) else (np.nan, np.nan)
    m = mod[(mod.case == case) & (mod.bits == bits)].n_star_model
    qm = iqr(m) if len(m) else (np.nan, np.nan)
    rows.append({"case": case, "bits": bits, "tol": tol, "n_star_emp_median": float(v.median()) if len(v) else np.nan, "emp_q25": q[0], "emp_q75": q[1], "emp_ics_with_plateau": len(v), "ics": len(d),
                 "censored_frac_at_max_n": d.censored_at_max_n.mean(), "n_star_model_median": float(m.median()) if len(m) else np.nan, "model_q25": qm[0], "model_q75": qm[1]})
cs = pd.DataFrame(rows)
save_table(cs, "e2_crossover_summary")
law = []
for case, d in cs[cs.tol == 1e-6].groupby("case"):
    v = d.dropna(subset=["n_star_model_median"])
    if len(v) >= 2:
        a, b = np.polyfit(v.bits, v.n_star_model_median, 1)
        slope, icpt = local_fit(loc, case)
        # model_bits_per_order shares its local fit with predicted_bits_per_order, so only the empirical fit tests the law.
        e = d[(d.emp_ics_with_plateau >= 0.5 * d.ics) & d.n_star_emp_median.notna()]
        emp = 1 / np.polyfit(e.bits, e.n_star_emp_median, 1)[0] if len(e) >= 3 else np.nan
        law.append({"case": case, "orders_per_bit": a, "model_bits_per_order": 1 / a, "intercept": b, "predicted_bits_per_order": (-slope) / np.log10(2), "empirical_bits_per_order": emp, "empirical_points": len(e), "rho_over_h": 10 ** (-slope)})
save_table(pd.DataFrame(law), "e2_crossover_law")
print(pd.DataFrame(law).to_string())
print(cs[cs.tol == 1e-6][["case", "bits", "n_star_emp_median", "emp_ics_with_plateau", "ics", "censored_frac_at_max_n", "n_star_model_median", "model_q25", "model_q75"]].to_string())
