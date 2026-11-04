"""E7: sensitivity to (n, p) vs FTLE over the initial-condition grid (RQ5). Spearman correlation and separation at lambda > 0.01."""
import numpy as np
import pandas as pd
from common import kind_runs, save_table
from scipy.stats import spearmanr

rows = []
for d, m in kind_runs("E7", "indicator"):
    if m["status"] != "completed":
        continue
    rows.append({"y": m["y"], "py": m["py"], "ftle": m["ftle"], "sens_n16": m["sensitivity"][0], "sens_n10": m["sensitivity"][1]})
df = pd.DataFrame(rows)
if len(df):
    save_table(df, "e7_indicator_grid", md_rows=20)
    out = []
    for col in ("sens_n16", "sens_n10"):
        r = spearmanr(df.ftle, df[col])
        chaotic = df.ftle > 0.01
        # AUC via rank statistic
        a, b = df[col][chaotic].values, df[col][~chaotic].values
        auc = float((a[:, None] > b[None, :]).mean()) if len(a) and len(b) else np.nan
        out.append({"indicator": col, "spearman_vs_ftle": r.statistic, "p": r.pvalue, "auc_chaotic_vs_regular": auc, "points": len(df), "chaotic_frac": float(chaotic.mean())})
    save_table(pd.DataFrame(out), "e7_indicator_summary")
    print(pd.DataFrame(out).to_string())
