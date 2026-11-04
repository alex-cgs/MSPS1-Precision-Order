"""Pilot regression (spec §10.4) through the new engine. Uses the cached reference when present."""
import numpy as np
import pytest

from msps.models import fit_local_law
from msps.runner.jobs import run_local, run_traj


@pytest.mark.slow
def test_pilot_regression():
    meta, data, _ = run_local({"case": "nb3_pilot", "ic": 0, "orders": [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36], "states": 60})
    ns = sorted(meta["median"])
    slope, _ = fit_local_law(ns, [meta["median"][n] for n in ns])
    assert abs(slope + 1.07) <= 0.03
    base = {"case": "nb3_pilot", "ic": 0, "method": "taylor", "schedule": "shared", "stop_err": 1.0}
    m16f, _, _ = run_traj({**base, "n": 16, "arith": "f64"})
    m24f, _, _ = run_traj({**base, "n": 24, "arith": "f64"})
    assert abs(m16f["tpred"]["1e-06"]["tpred"] - 70.4) < 0.06
    assert abs(m24f["tpred"]["1e-06"]["tpred"] - 70.4) < 0.06
    m16, _, _ = run_traj({**base, "n": 16, "arith": "mpfr-80"})
    assert abs(m16["tpred"]["1e-06"]["tpred"] - 79.1) < 0.06


@pytest.mark.slow
def test_exact_growth_near_33():
    from msps.runner.jobs import run_exact
    meta, _, _ = run_exact({"case": "nb3_pilot", "ic": 0, "n": 8, "max_steps": 3, "max_wall": 120, "max_bits": 10**8})
    assert all(25 < g < 40 for g in meta["growth"][1:])
    assert np.isclose(meta["predicted_factor"], 33)
