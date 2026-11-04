"""E0: pass/fail table, spec §5 regression targets next to the values from the new code."""
import math

import numpy as np
import pandas as pd
from common import kind_runs, load_data, local_table, save_table, traj_rows
from model_tools import obs_one_curve
from msps.models import crossover, fit_constants, fit_local_law, predicted_tpred

rows = []


def chk(name, target, value, ok, note=""):
    rows.append({"check": name, "target": target, "value": value, "pass": bool(ok), "note": note})


T = 120.0
for d, m in kind_runs("E0", "reference"):
    r = m["reference"]
    if m["config"]["case"] == "nb3_pilot":
        chk("pilot reference steps", 2605, r["steps"], r["steps"] == 2605)
        chk("pilot h min", 4.8e-7, r["hmin"], abs(r["hmin"] - 4.8e-7) < 5e-8)
        chk("pilot reference check", 1.7e-41, r["max_pos_diff"], abs(math.log10(r["max_pos_diff"]) + 40.8) < 0.1)
        chk("closest approach", 4.1e-4, r["dmin"], abs(r["dmin"] - 4.1e-4) < 5e-5)
    else:
        chk(f"reference check {m['config']['case']}", "<1e-30", r["max_pos_diff"], r["check_passed"])
amp = {}
for d, m in kind_runs("E0", "amplification"):
    a = load_data(d)
    amp[(m["config"]["case"], m["config"]["ic"])] = a
    chk("amplification A(T)", 7.8e8, m["A_T"], abs(m["A_T"] / 7.8e8 - 1) < 0.02)
    chk("lambda fit", 0.16, m["lambda_fit"], abs(m["lambda_fit"] - 0.16) < 0.01)
local = local_table("E0")
med = local.groupby("n")["log10_eps"].median()
slope, icpt = fit_local_law(med.index.values, med.values)
chk("local slope", "-1.07 +- 0.03", slope, abs(slope + 1.07) <= 0.03)
chk("local intercept", -2.29, icpt, abs(icpt + 2.29) < 0.05)
chk("rho/h effective", 11.7, 10 ** (-slope), abs(10 ** (-slope) - 11.7) < 0.3)
tr = traj_rows("E0")
tr = tr[tr.sample_all & (tr.tol == 1e-6)]
sel = tr.set_index(["arith", "n"])["tpred"]
for n, target in {4: 0.5, 6: 2.4, 8: 2.7, 10: 11.3, 12: 31.3, 14: 57.5, 16: 70.4, 18: 70.4, 20: 70.4, 24: 70.4, 28: 70.4}.items():
    chk(f"f64 Tpred n={n}", target, sel[("f64", n)], abs(sel[("f64", n)] - target) < 0.06)
for n, target in {12: 31.3, 14: 57.6, 16: 79.1, 18: 118.8}.items():
    for a in ("mpfr-80", "mpfr-113", "mpfr-160"):
        chk(f"{a} Tpred n={n}", target, sel[(a, n)], abs(sel[(a, n)] - target) < 0.06)
for n in (20, 24, 28):
    v = tr[(tr.n == n) & (tr.arith == "mpfr-80")].iloc[0]
    chk(f"mpfr-80 censored n={n}", ">120", v.tpred, bool(v.censored))
ob = obs_one_curve(tr, amp, T, stride=10)
fit = fit_constants(ob[:, :3], slope, icpt)
chk("c_t", 7.43, fit["c_t"], abs(fit["c_t"] - 7.43) < 0.05)
chk("c_r", 4.21, fit["c_r"], abs(fit["c_r"] - 4.21) < 0.05)
chk("fit rms", 0.60, fit["rms"], abs(fit["rms"] - 0.60) < 0.03)
chk("fit obs", 6207, fit["n_obs"], abs(fit["n_obs"] - 6207) < 150, "coarse sampling differs by rounding of thin-out")
for bits, target in {53: 14.8, 80: 22.5, 113: 31.8, 160: 45.0}.items():
    v = crossover(fit["c_t"], fit["c_r"], slope, icpt, bits)
    chk(f"n* {bits} bits", target, float(v), abs(v - target) < 0.2)
a = amp[("nb3_pilot", 0)]
for n, target in {10: 4.3, 12: 27.2, 14: 40.4, 16: 79.0, 18: 118.8}.items():
    v, _ = predicted_tpred(a["t"].values, np.log10(a["A"].values), fit["c_t"], fit["c_r"], slope, icpt, n, 113, 1e-6)
    chk(f"model Tpred n={n}", target, v, abs(v - target) < 0.5)
for d, m in kind_runs("E0", "exact"):
    chk("exact bits step 4", 7541120, m["bits"][4], m["bits"][4] == 7541120)
    chk("exact growth factors", "33.6, 31.7, 31.6", str([round(g, 1) for g in m["growth"][1:]]), abs(m["growth"][1] - 33.6) < 0.1)
for d, m in kind_runs("E0", "kepler"):
    c = m["config"]
    if c["n"] == 24 and c["arith"] == "mpfr-113":
        chk(f"Kepler oracle {c['case']} n=24", "<1e-15", m["E_max"], m["E_max"] < 1e-15)
for d, m in kind_runs("E0", "closure"):
    chk("figure-eight closure at P=6.32591398", "<1e-6 (8-digit ICs)", m["pos_diff"], m["pos_diff"] < 1e-6, "verifies period and IC digits")
for d, m in kind_runs("E0", "stats"):
    name = m["config"]["case"]
    target = 0.9056 if "lorenz" in name else 0.0714
    chk(f"lambda {name}", target, m["lambda_est"], abs(m["lambda_est"] - target) < 0.02, "Sprott 2003 value, high-precision run here")
reg = {"duffing:gamma_0_20": (1, False), "duffing:gamma_0_28": (2, False), "duffing:gamma_0_29": (4, False), "duffing:gamma_0_37": (5, False),
       "duffing:gamma_0_50": (None, True), "duffing:gamma_0_65": (2, False)}
for d, m in kind_runs("E0", "strobe"):
    c = m["config"]["case"]
    per, chaos = reg[c]
    ok = m["lambda_est"] > 0.05 if chaos else m["distinct_1e-6"] == per
    chk(f"{c} ic={m['config']['ic']}", "chaotic" if chaos else f"period {per}", f"distinct={m['distinct_1e-6']} lambda={m['lambda_est']:.4f}", ok)
df = pd.DataFrame(rows)
save_table(df, "e0_validation")
print(f"E0 {df['pass'].sum()}/{len(df)} checks pass")
print(df[~df["pass"]].to_string())
