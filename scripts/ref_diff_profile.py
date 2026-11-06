"""Replays a stored reference's check and records when the difference crosses tol_check.

usage: ref_diff_profile.py CASE IC [STOP_DIFF]. Appends to results/derived/e11_ref_diff_profile.csv: one row per decade the running maximum crosses.
"""
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msps.arith import get_arith  # noqa: E402
from msps.diagnostics import error_positions  # noqa: E402
from msps.integrators.taylor import TaylorStepper  # noqa: E402
from msps.runner.references import _load, ref_dir  # noqa: E402
from msps.systems import load_case  # noqa: E402
import pickle  # noqa: E402

case_id, ic = sys.argv[1], int(sys.argv[2])
stop = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-15
case = load_case(case_id)
p = case.card["reference"]
with open(ref_dir(case, ic) / "ref.pkl", "rb") as f:
    raw = pickle.load(f)
get_arith(f"mpfr-{p['bits']}").activate()
states = raw["states"]
ar = get_arith(f"mpfr-{p['check_bits']}")
ar.activate()
st = TaylorStepper(case, p["check_order"], ar)
z = [ar.convert(v) for v in case.ic(ic)]
idx = case.system.positions
out = ROOT / "results" / "derived" / "e11_ref_diff_profile.csv"
new = not out.exists()
t, top, decade, t0 = 0.0, 0.0, None, time.time()
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["case", "ic", "order", "bits", "check_order", "check_bits", "step", "t", "max_diff", "h", "wall"])
    for k, h in enumerate(raw["steps"]):
        z = st.horner(st.coeffs(z), ar.convert(h))
        t += float(h)
        d = error_positions(ar, z, _load(states[k + 1]), idx)
        top = max(top, d)
        dec = math.floor(math.log10(top)) if 0 < top < math.inf else (999 if top else None)
        if dec is not None and dec != decade:
            decade = dec
            w.writerow([case_id, ic, p["order"], p["bits"], p["check_order"], p["check_bits"], k + 1, t, top, float(h), round(time.time() - t0)])
            f.flush()
        if top > stop:
            break
    else:
        w.writerow([case_id, ic, p["order"], p["bits"], p["check_order"], p["check_bits"], len(raw["steps"]), t, top, float(raw["steps"][-1]), round(time.time() - t0)])
