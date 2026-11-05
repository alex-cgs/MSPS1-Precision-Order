"""E8: method families on a common footing (RQ7): T_pred, invariants, wall time, RHS evaluations."""
import numpy as np
import pandas as pd
from common import load_data, save_table, traj_rows

tr = traj_rows("E8")
rows = []
for r in tr[tr.tol == 1e-6].itertuples():
    d = load_data(r.run)
    inv = [c for c in d.columns if c.startswith("d") and c != "dC"]
    cols = {f"max_{c}": float(np.nanmax(d[c].values)) for c in inv}
    if "dC" in d:
        cols["max_dC"] = float(np.nanmax(d["dC"].values))
    last_e = float(d["E"].values[-1])
    label = r.method
    param = {"taylor": f"n={r.n}", "rk4": f"m={r.substeps}", "dop853": f"m={r.substeps}" if r.schedule == "shared" else f"rtol={r.rtol}", "gl": f"s={r.s},m={r.substeps},tol={r.solver_tol}", "yoshida": f"k={r.k},m={r.substeps}"}[label]
    rows.append({"case": r.case, "method": label, "param": param, "arith": r.arith, "schedule": r.schedule, "tpred_1e-6": r.tpred, "censored": r.censored, "wall": r.wall, "steps": r.steps,
                 "t_end": float(d["t"].values[-1]), "status": r.status, "E_last": last_e, "solver_iterations": r.solver_iterations, "solver_failures": r.solver_failures, **cols})
df = pd.DataFrame(rows).sort_values(["case", "method", "arith", "wall"])
save_table(df, "e8_methods")
print(df.groupby(["case", "method", "arith"]).agg(best_tpred=("tpred_1e-6", "max"), min_wall=("wall", "min"), max_wall=("wall", "max"), runs=("wall", "size")).to_string())
