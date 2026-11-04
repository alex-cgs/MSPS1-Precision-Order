"""E3: fit on half the ICs, test on the other half; one-curve model vs FTLE-based (time-resolved) model (spec §4.4, RQ2/H2)."""
import numpy as np
import pandas as pd
from common import amp_table, local_table, save_table, traj_rows
from model_tools import fit_constants, fit_resolved, local_fit, obs_one_curve, obs_resolved, one_curve_pred, resolved_pred
from msps.models import predicted_tpred, resolved_logE  # noqa: F401
from msps.systems import load_case

STUDY = "E2"
tr = traj_rows(STUDY)
tr = tr[(tr.schedule == "shared") & (tr.tol == 1e-6)]
tr3 = traj_rows(STUDY)
tr3 = tr3[tr3.schedule == "shared"]
amp = amp_table(STUDY)
loc = local_table(STUDY)
rows, pts, tp_rows = [], [], []
for case, d in tr.groupby("case"):
    ics = sorted(d.ic.unique())
    ics = [i for i in ics if (case, i) in amp]
    if len(ics) < 4:
        continue
    T = float(load_case(case).card["T"])
    train, test = ics[0::2], ics[1::2]
    slope, icpt = local_fit(loc, case, train)
    dtr, dte = d[d.ic.isin(train)], d[d.ic.isin(test)]
    ob_tr, ob_te = obs_one_curve(dtr, amp, T), obs_one_curve(dte, amp, T)
    if len(ob_tr) < 20 or len(ob_te) < 20:
        continue
    fit = fit_constants(ob_tr[:, :3], slope, icpt)
    res = ob_te[:, 2] - one_curve_pred(fit, slope, icpt, ob_te)
    rr_tr, rr_te = obs_resolved(dtr, amp, T), obs_resolved(dte, amp, T)
    fr = fit_resolved(rr_tr)
    res_r = rr_te[:, 0] - resolved_pred((fr["c_t"], fr["c_r"]), rr_te)
    row = {"case": case, "train_ics": len(train), "test_ics": len(test), "slope": slope, "log10_K": icpt,
           "c_t": fit["c_t"], "c_r": fit["c_r"], "train_rms": fit["rms"], "test_rms": float(np.sqrt(np.mean(res**2))), "test_within_1_decade": float(np.mean(np.abs(res) < 1)),
           "test_median_abs": float(np.median(np.abs(res))), "ftle_c_t": fr["c_t"], "ftle_c_r": fr["c_r"], "ftle_train_rms": fr["rms"], "ftle_test_rms": float(np.sqrt(np.mean(res_r**2))),
           "ftle_test_within_1_decade": float(np.mean(np.abs(res_r) < 1)), "ftle_test_median_abs": float(np.median(np.abs(res_r))), "test_points": len(ob_te)}
    rows.append(row)
    # predicted vs measured log E at the last stored time of each test run below 1e-2, plus T_pred out of sample
    for ic in test:
        a = amp[(case, ic)]
        tt, logA = a["t"].values, np.log10(a["A"].values)
        for r in dte[dte.ic == ic].itertuples():
            for tol in (1e-6, 1e-3):
                rec = tr3[(tr3.run == r.run) & (tr3.tol == tol)].iloc[0]
                tp, cens = predicted_tpred(tt, logA, fit["c_t"], fit["c_r"], slope, icpt, r.n, r.bits, tol)
                tp_rows.append({"case": case, "ic": ic, "n": r.n, "bits": r.bits, "tol": tol, "tpred": rec.tpred, "censored": rec.censored, "tpred_model": tp, "model_censored": cens})
    idx = np.random.default_rng(0).permutation(len(ob_te))[:3000]
    for i in idx:
        pts.append({"case": case, "n": ob_te[i, 0], "bits": ob_te[i, 1], "ic": ob_te[i, 3], "t": ob_te[i, 4],
                    "log10E_measured_minus_logA": ob_te[i, 2], "log10E_pred_minus_logA": float(one_curve_pred(fit, slope, icpt, ob_te[[i]])[0])})
out = pd.DataFrame(rows)
save_table(out, "e3_model_out_of_sample")
save_table(pd.DataFrame(tp_rows), "e3_tpred_out_of_sample", md_rows=30)
save_table(pd.DataFrame(pts), "e3_points_sample", md_rows=10)
tp = pd.DataFrame(tp_rows)
if len(tp):
    un = tp[~tp.censored & ~tp.model_censored]
    un = un.assign(rel=np.abs(un.tpred_model - un.tpred) / np.maximum(un.tpred, 1e-9))
    g = un.groupby(["case", "tol"]).agg(median_rel_err=("rel", "median"), q75_rel_err=("rel", lambda x: np.percentile(x, 75)), runs=("rel", "size")).reset_index()
    save_table(g, "e3_tpred_accuracy")
    print(g.to_string())
print(out.to_string())
