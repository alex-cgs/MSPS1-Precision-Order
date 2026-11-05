"""E4 flag: Hénon-Heiles grows slower than (d-1)n+1. Exact degree of its order-n Taylor map in the state, against the stored growth."""
import pandas as pd
import sympy as sp
from common import ROOT, save_table

x, y, u, v = S = sp.symbols("x y u v")
F = (u, v, -x - 2 * x * y, -y - x**2 + y**2)
cur, deg, rows = list(S), [1], []
for k in range(1, 13):
    cur = [sp.expand(sum(sp.diff(c, s) * f for s, f in zip(S, F))) for c in cur]
    deg.append(max(deg[-1], max(sp.Poly(c, *S).total_degree() for c in cur)))
g = pd.read_csv(ROOT / "results" / "derived" / "e4_exact_growth.csv")
g = g[g.case == "henon_heiles:rational_ic"]
for r in g.itertuples():
    rows.append({"n": int(r.n), "map_degree": deg[int(r.n)], "law_(d-1)n+1": int(r.predicted_factor), "measured_growth": r.growth_mean_after_first})
save_table(pd.DataFrame(rows).sort_values("n"), "e4_hh_map_degree")
print(pd.DataFrame(rows).sort_values("n").to_string(index=False))
