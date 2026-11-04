"""E4: exact-arithmetic bit growth against (d-1)n+1 (RQ9). Growth from step 2 on, the first step reflects IC size."""
import numpy as np
import pandas as pd
from common import kind_runs, save_table

rows = []
for kind in ("exact", "exact_map"):
    for d, m in kind_runs("E4", kind):
        if m.get("status") == "skipped_irrational_ic":
            continue
        c = m["config"]
        g = m["growth"][1:]
        rows.append({"case": c["case"], "params": str(c.get("params", "")), "ic": c.get("ic"), "n": c.get("n"), "degree": m["degree"], "predicted_factor": m["predicted_factor"],
                     "steps_done": len(m["growth"]), "growth_mean_after_first": float(np.mean(g)) if g else np.nan, "growth_last": m["growth"][-1], "bits_last": m["bits"][-1], "wall_last": m["walls"][-1],
                     "bits": str(m["bits"][:12]), "ratio_vs_predicted": (float(np.mean(g)) / m["predicted_factor"]) if g else np.nan})
df = pd.DataFrame(rows).sort_values(["case", "params", "n", "ic"], na_position="last")
save_table(df, "e4_exact_growth")
print(df[["case", "params", "ic", "n", "degree", "predicted_factor", "steps_done", "growth_mean_after_first", "ratio_vs_predicted", "bits_last", "wall_last"]].to_string())
