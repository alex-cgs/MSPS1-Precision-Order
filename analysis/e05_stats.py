"""E5: statistics vs trajectories (RQ3). Reference ensemble = the highest (bits, n) in each case's grid; noise floor = split-half W1 of that ensemble."""
import numpy as np
import pandas as pd
from common import bits_of, kind_runs, load_data, save_table, sim_t_end
from msps.diagnostics import ks_escape, wasserstein1_hist

rows = []
by_case = {}
for d, m in kind_runs("E5", "stats"):
    c = m["config"]
    by_case.setdefault(c["case"], []).append((d, m))
out, run_rows = [], []
for case, runs in by_case.items():
    obs = list(runs[0][1]["config"]["observables"])
    groups = {}
    for d, m in runs:
        c = m["config"]
        groups.setdefault((c["arith"], c["n"]), []).append((d, m))
    ref_key = max(groups, key=lambda k: (bits_of(k[0]), k[1]))
    ens = {}
    for key, items in groups.items():
        hist = {o: {} for o in obs}
        for d, m in items:
            df = load_data(d)
            for o in obs:
                cnt = df[f"{o}_count"].dropna().values
                edges = np.append(df[f"{o}_left"].dropna().values, df[f"{o}_right"].dropna().values[-1])
                hist[o][m["config"]["ic"]] = (cnt, edges)
        ens[key] = hist
    ref = ens[ref_key]
    ref_lam = np.nanmedian([m["lambda_est"] for _, m in groups[ref_key]])
    floor = {}
    for o in obs:
        ic = sorted(ref[o])
        a = sum(ref[o][i][0] for i in ic[0::2])
        b = sum(ref[o][i][0] for i in ic[1::2])
        floor[o] = wasserstein1_hist(a, b, ref[o][ic[0]][1]) if len(ic) >= 2 and a.sum() and b.sum() else np.nan
    ref_esc = [m["escape_time"] if m["escape_time"] is not None else np.nan for _, m in groups[ref_key]]
    ref_bodies = np.array([m["escape_body"] for _, m in groups[ref_key] if m["escape_body"] is not None])
    for key, items in groups.items():
        arith, n = key
        lams = np.array([m["lambda_est"] for _, m in items], float)
        tp6 = [m["tpred"]["1e-06"]["tpred"] for _, m in items if m["tpred"]]
        cens6 = [m["tpred"]["1e-06"]["censored"] for _, m in items if m["tpred"]]
        tp3 = [m["tpred"]["0.001"]["tpred"] for _, m in items if m["tpred"]]
        status = pd.Series([m["status"] for _, m in items]).value_counts().to_dict()
        row = {"case": case, "arith": arith, "bits": bits_of(arith), "n": n, "runs": len(items), "lambda_median": float(np.nanmedian(lams)), "lambda_q25": float(np.nanpercentile(lams, 25)),
               "lambda_q75": float(np.nanpercentile(lams, 75)), "lambda_ref": float(ref_lam), "lambda_abs_err": float(abs(np.nanmedian(lams) - ref_lam)),
               "tpred_median_1e-6": float(np.median(tp6)) if tp6 else np.nan, "censored_frac_1e-6": float(np.mean(cens6)) if cens6 else np.nan, "tpred_median_1e-3": float(np.median(tp3)) if tp3 else np.nan,
               "status": str(status), "wall_median": float(np.median([m["wall"] for _, m in items])), "steps_median": float(np.median([m["steps"] for _, m in items])),
               "t_end_median": float(np.nanmedian([sim_t_end(m) for _, m in items])), "is_reference": key == ref_key}
        for o in obs:
            ics = sorted(ens[key][o])
            a = sum(ens[key][o][i][0] for i in ics)
            b = sum(ref[o][i][0] for i in sorted(ref[o]))
            edges = ref[o][sorted(ref[o])[0]][1]
            row[f"W1_{o}"] = wasserstein1_hist(a, b, edges) if a.sum() and b.sum() else np.nan
            row[f"W1_floor_{o}"] = floor[o]
        esc = [m["escape_time"] if m["escape_time"] is not None else np.nan for _, m in items]
        if any(m["escape_body"] is not None for _, m in items) or len(ref_bodies):
            ks = ks_escape(esc, ref_esc)
            row.update({"esc_frac": float(np.mean(np.isfinite(esc))), "esc_frac_ref": float(np.mean(np.isfinite(ref_esc))), "esc_time_median": float(np.nanmedian(esc)) if np.isfinite(esc).any() else np.nan,
                        "esc_time_median_ref": float(np.nanmedian(ref_esc)) if np.isfinite(ref_esc).any() else np.nan, "esc_ks": ks["ks"], "esc_ks_p": ks["p"], "esc_na": ks["na"], "esc_nb": ks["nb"]})
            bodies = np.array([m["escape_body"] for _, m in items if m["escape_body"] is not None])
            if bodies.size and ref_bodies.size:
                p = np.bincount(bodies, minlength=3) / bodies.size
                q = np.bincount(ref_bodies, minlength=3) / ref_bodies.size
                row["esc_body_TV"] = float(0.5 * np.abs(p - q).sum())
        out.append(row)
df = pd.DataFrame(out).sort_values(["case", "bits", "n"])
save_table(df, "e5_stats_summary")
keep = ["case", "arith", "n", "runs", "lambda_median", "lambda_abs_err", "tpred_median_1e-6", "censored_frac_1e-6", "t_end_median", "status"]
print(df[[c for c in keep if c in df] + [c for c in df if c.startswith("W1_") and "floor" not in c]].to_string())
