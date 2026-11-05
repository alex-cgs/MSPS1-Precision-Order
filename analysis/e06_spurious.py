"""E6: fake periodicity, bifurcation shifts, section survival (RQ4)."""
import numpy as np
import pandas as pd
from common import kind_runs, load_data, save_table

rows = []
for d, m in kind_runs("E6", "map"):
    c = m["config"]
    rows.append({"case": c["case"], "arith": c["arith"], "ic": c["ic"], "period": m["period"], "transient": m["transient"], "lambda_est": m["lambda_est"]})
df = pd.DataFrame(rows)
if len(df):
    g = df.groupby(["case", "arith"]).agg(period_median=("period", "median"), period_q25=("period", lambda x: np.nanpercentile(x, 25)), period_q75=("period", lambda x: np.nanpercentile(x, 75)),
                                          period_max=("period", "max"), transient_median=("transient", "median"), no_period_found=("period", lambda x: x.isna().mean()), runs=("period", "size"),
                                          lambda_median=("lambda_est", "median")).reset_index()
    save_table(g, "e6_map_periodicity")
    print(g.to_string())
rows = []
for d, m in kind_runs("E6", "map_diagram"):
    c = m["config"]
    t = load_data(d)
    mu, lam = t["mu"].values, t["lambda"].values
    chaotic = lam > 0.01
    rng = float(mu[np.argmax(chaotic)]) if chaotic.any() else np.nan
    bound = t["bounded"].values.astype(bool)
    rows.append({"builder": c["builder"], "w": c.get("w"), "arith": c["arith"], "onset_mu_lambda_pos": rng, "frac_mu_chaotic": float(chaotic.mean()), "frac_mu_bounded": float(bound.mean()), "lambda_max": float(np.nanmax(lam[bound])) if bound.any() else np.nan, "run": str(d)})
dg_ = pd.DataFrame(rows)
if len(dg_):
    dg_ = dg_.assign(w=dg_.w.astype(object)).sort_values(["builder", "arith"], kind="stable")
    save_table(dg_.drop(columns="run"), "e6_map_diagrams")
    print(dg_.drop(columns="run").to_string())
rows = []
for d, m in kind_runs("E6", "strobe"):
    c = m["config"]
    rows.append({"case": c["case"], "n": c["n"], "arith": c["arith"], "ic": c["ic"], "distinct": m["distinct_1e-6"], "lambda_est": m["lambda_est"], "status": m["status"]})
df = pd.DataFrame(rows)
if len(df):
    g = df.groupby(["case", "n", "arith"]).agg(distinct_median=("distinct", "median"), lambda_median=("lambda_est", "median"), blown=("status", lambda x: (x != "completed").mean())).reset_index()
    ref = g[(g.n == 16) & (g.arith == "f64")].set_index("case")
    g["regime_changed"] = [(row.distinct_median != ref.loc[row.case, "distinct_median"]) or ((row.lambda_median > 0.02) != (ref.loc[row.case, "lambda_median"] > 0.02)) for row in g.itertuples()]
    # distinct counts also rise for a slowly drifting fixed point; only a sign change of lambda is a change of dynamics
    g["spurious_chaos"] = [(row.lambda_median > 0.02) and not (ref.loc[row.case, "lambda_median"] > 0.02) for row in g.itertuples()]
    g["spurious_order"] = [(row.lambda_median <= 0.02) and (ref.loc[row.case, "lambda_median"] > 0.02) for row in g.itertuples()]
    save_table(g, "e6_duffing_regimes")
    print(g[g.spurious_chaos | g.spurious_order].to_string())
rows = []
secs = {}
for d, m in kind_runs("E6", "poincare"):
    c = m["config"]
    t = load_data(d)
    secs[(c["case"], c["n"], c["arith"], c["ic"])] = t
    rows.append({"case": c["case"], "n": c["n"], "arith": c["arith"], "ic": c["ic"], "crossings": m["crossings"], "max_dH": m["max_dH"], "status": m["status"], "wall": m["wall"]})
df = pd.DataFrame(rows)
if len(df):
    edges_y, edges_p = np.linspace(-1, 1.5, 80), np.linspace(-1, 1, 64)
    occ = []
    for (case, n, arith, ic), t in secs.items():
        ref = secs.get((case, 24, "dd", ic))
        if ref is None or len(ref) == 0 or len(t) == 0:
            occ.append(np.nan)
            continue
        H, _, _ = np.histogram2d(t.y, t.py, [edges_y, edges_p])
        R, _, _ = np.histogram2d(ref.y, ref.py, [edges_y, edges_p])
        a, b = H > 0, R > 0
        occ.append(float((a & b).sum() / (a | b).sum()))
    df["section_jaccard"] = occ
    g = df.groupby(["case", "n", "arith"]).agg(jaccard_median=("section_jaccard", "median"), max_dH_median=("max_dH", "median"), crossings_median=("crossings", "median"),
                                               blown=("status", lambda x: (x != "completed").mean()), runs=("status", "size")).reset_index()
    save_table(g, "e6_henon_sections")
    print(g.to_string())
