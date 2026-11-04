"""E1: local truncation law per system: slope, rho/h, log10 K; medians and IQR per order (spec §4.7: >= 60 states)."""
import numpy as np
import pandas as pd
from common import local_table, save_table
from msps.models import fit_local_law

loc = local_table("E1")
rows, per_n = [], []
for case, d in loc.groupby("case"):
    per = d.groupby("n")["log10_eps"].agg(median="median", q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75), states="count").reset_index()
    per.insert(0, "case", case)
    per_n.append(per)
    slope, icpt = fit_local_law(per.n.values, per["median"].values)
    res = per["median"].values - (icpt + slope * (per.n.values + 1))
    rows.append({"case": case, "slope": slope, "log10_K": icpt, "rho_over_h": 10 ** (-slope), "rms_resid": float(np.sqrt(np.mean(res[per["median"] > -110] ** 2))),
                 "h_median": float(d.h.median()), "ics": d.ic.nunique()})
save_table(pd.DataFrame(rows), "e1_local_law")
save_table(pd.concat(per_n), "e1_local_medians")
print(pd.DataFrame(rows).to_string())
