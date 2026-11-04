"""E10: throughput, cost per accurate time unit, bitwise reproducibility across devices (RQ8)."""
import numpy as np
import pandas as pd
from common import save_table
from msps.io import iter_runs

rows, acc = [], []
for d, m in iter_runs("E10"):
    for r in m.get("rows", []):
        rows.append({"label": m["label"], "size": m["config"]["size"], **{k: r.get(k) for k in ("case", "n", "prec", "batch", "steps", "repeats", "wall_median", "compile_s", "throughput_median", "peak_bytes", "device", "fingerprint", "error")},
                     "thr_q25": (r.get("throughput_iqr") or [np.nan])[0], "thr_q75": (r.get("throughput_iqr") or [np.nan, np.nan])[-1]})
    for a in m.get("accuracy", []):
        acc.append({"label": m["label"], "size": m["config"]["size"], **a})
df = pd.DataFrame(rows)
if len(df):
    save_table(df, "e10_throughput", md_rows=60)
    fp = df.dropna(subset=["fingerprint"]).groupby(["case", "n", "prec", "batch", "steps"]).agg(devices=("label", "nunique"), distinct_fingerprints=("fingerprint", "nunique"), labels=("label", lambda x: ",".join(sorted(set(x))))).reset_index()
    fp = fp[fp.devices > 1]
    save_table(fp, "e10_reproducibility")
    print(fp.to_string())
    print(df.groupby(["label", "case", "prec", "n"]).throughput_median.max().to_string())
if acc:
    a = pd.DataFrame(acc)
    save_table(a, "e10_accuracy")
    print(a.to_string())
