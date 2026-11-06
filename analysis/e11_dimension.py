"""E11: dimension scaling. T_pred and n* vs N, N-body and Lorenz-96."""
import numpy as np
import pandas as pd
from common import local_table, save_table, traj_rows
from model_tools import local_fit
from msps.models import empirical_crossover
from msps.systems import load_case

# E11hi: N = 8, 10 at T = 8, and the T = 30 bridge on one IC each. Their T_pred is censored at their own T, so the T column matters.
tr = pd.concat([traj_rows("E11"), traj_rows("E11hi")])
rows = []
loc = pd.concat([local_table("E11"), local_table("E11hi")])
for case, d in tr[tr.tol == 1e-6].groupby("case"):
    c = load_case(case)
    N = c.system.meta.get("N", c.system.dim)
    family = case.split(":")[0] + ("_" + case.split(":")[1].split("_")[0] if ":" in case and case.startswith("nbN_alternating") else "")
    slope, icpt = local_fit(loc, case) if case in set(loc.case) else (np.nan, np.nan)
    for arith, da in d.groupby("arith"):
        by_ic = []
        for ic, di in da.groupby("ic"):
            di = di.sort_values("n")
            tp = di.tpred.values
            plateau = len(tp) >= 3 and not di.censored.values[-1] and abs(tp[-1] - tp[-3]) <= 0.05 * tp[-1]
            by_ic.append(empirical_crossover(di.n.values, tp) if plateau else np.nan)
        top = da[da.n == da.n.max()]
        rows.append({"case": case, "family": family, "N": N, "T": float(c.card["T"]), "dim": c.system.dim, "tape_mul": c.tape.n_mul, "arith": arith, "bits": da.bits.iloc[0], "tpred_at_max_n": float(top.tpred.median()), "censored_frac_at_max_n": float(top.censored.mean()),
                     "n_star_empirical": float(np.nanmedian(by_ic)) if np.isfinite(by_ic).any() else np.nan, "ics_with_plateau": int(np.isfinite(by_ic).sum()), "ics": len(by_ic), "slope": slope, "rho_over_h": 10 ** (-slope) if slope == slope else np.nan, "wall_median_at_max_n": float(top.wall.median())})
df = pd.DataFrame(rows).sort_values(["family", "N", "T", "bits"])
save_table(df, "e11_dimension")
print(df.to_string())
