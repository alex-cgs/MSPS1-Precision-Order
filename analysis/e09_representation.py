"""E9: representation vs bit count (RQ6)."""
import numpy as np
import pandas as pd
from common import bits_of, kind_runs, save_table, traj_rows

rows = []
for d, m in kind_runs("E9", "map"):
    c = m["config"]
    rows.append({"case": c["case"], "arith": c["arith"], "bits": bits_of(c["arith"]), "ic": c["ic"], "k_cross": m["k_cross"], "censored": m["censored"], "period": m["period"], "lambda_est": m["lambda_est"], "wall": m["wall"]})
mp = pd.DataFrame(rows)
if len(mp):
    g = mp.groupby(["case", "bits", "arith"]).agg(k_cross_median=("k_cross", "median"), k_q25=("k_cross", lambda x: np.nanpercentile(x, 25)), k_q75=("k_cross", lambda x: np.nanpercentile(x, 75)),
                                                  censored=("censored", "mean"), lambda_median=("lambda_est", "median"), wall_median=("wall", "median"), runs=("wall", "size")).reset_index()
    save_table(g, "e9_maps")
    print(g.to_string())
tr = traj_rows("E9")
if len(tr):
    s = tr.groupby(["case", "n", "arith", "bits", "tol"]).agg(tpred_median=("tpred", "median"), tq25=("tpred", lambda x: np.percentile(x, 25)), tq75=("tpred", lambda x: np.percentile(x, 75)), censored=("censored", "mean"),
                                                              wall_median=("wall", "median"), runs=("wall", "size"), statuses=("status", lambda x: str(x.value_counts().to_dict()))).reset_index()
    save_table(s, "e9_odes")
    print(s[s.tol == 1e-6].to_string())
