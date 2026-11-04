"""GPU kernels against their CPU reference and against the scalar high-precision path (CLAUDE.md workflow rule)."""
from fractions import Fraction

import numpy as np
import pytest

from msps.arith import get_arith
from msps.gpu.kernels import BatchedTaylor, compare_states
from msps.integrators import TaylorStepper
from msps.systems import load_case

CASES = ["lorenz63", "nb3_pilot"]
TOL = {"f32": 1e-4, "f64": 1e-12, "dd": 1e-27}


def _ics(case, k=4):
    return [case.ic(0)] + [[v * (1 + Fraction(j, 1000)) for v in case.ic(0)] for j in range(1, k)] if case.name == "lorenz63" else [case.ic(0)]


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("prec", ["f32", "f64", "dd"])
def test_numpy_reference_matches_scalar_taylor(spec, prec):
    case = load_case(spec)
    n = 12
    states = _ics(case)
    kern = BatchedTaylor(case.tape, n, prec, "numpy")
    z = kern.from_fractions(states)
    h = np.full(len(states), 1 / 64)
    out = kern.step(z, h)
    arr = out if prec != "dd" else out[0].astype(np.longdouble) + out[1]
    a = get_arith("mpfr-300")
    a.activate()
    st = TaylorStepper(case, n, a)
    for m, s in enumerate(states):
        r = st.step([a.convert(v) for v in s], a.convert(Fraction(1, 64)))
        rf = np.array([a.to_float(x) for x in r], dtype=np.longdouble)
        err = np.abs(arr[:, m].astype(np.longdouble) - rf).max() / np.abs(rf).max()
        assert err < max(TOL[prec], 1e-15), err  # rf holds float64 only; test_numpy_dd_accuracy_beyond_double checks dd


def test_numpy_dd_accuracy_beyond_double():
    case = load_case("lorenz63")
    kern = BatchedTaylor(case.tape, 16, "dd", "numpy")
    s = case.ic(0)
    z = kern.from_fractions([s])
    h = np.full(1, 1 / 64)
    z = kern.step(z, h)
    a = get_arith("mpfr-300")
    a.activate()
    st = TaylorStepper(case, 16, a)
    r = st.step([a.convert(v) for v in s], a.convert(Fraction(1, 64)))
    R = [a.to_fraction(v) for v in r]
    err = max(abs(Fraction(float(z[0][i, 0])) + Fraction(float(z[1][i, 0])) - R[i]) for i in range(3))
    assert float(err) < 1e-26


def test_adaptive_advance_lands_on_target_numpy():
    case = load_case("nb3_pilot")
    kern = BatchedTaylor(case.tape, 16, "f64", "numpy")
    z = kern.from_fractions([case.ic(0)])
    t = np.zeros(1)
    z, t, steps = kern.advance(z, t, 2.0)
    assert t[0] == 2.0 and steps[0] > 1


@pytest.mark.gpu
@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("prec", ["f32", "f64", "dd"])
def test_jax_matches_numpy(spec, prec):
    jax = pytest.importorskip("jax")
    case = load_case(spec)
    n = 12
    states = _ics(case)
    k_np = BatchedTaylor(case.tape, n, prec, "numpy")
    k_jx = BatchedTaylor(case.tape, n, prec, "jax")
    h = np.full(len(states), 1 / 64)
    a = k_np.step(k_np.from_fractions(states), h)
    b = k_jx.to_numpy(k_jx.step(k_jx.from_fractions(states), jax.numpy.asarray(h)))
    cmp = compare_states(a, b)
    print(spec, prec, jax.devices()[0], cmp)
    assert cmp["max_abs_diff"] < {"f32": 1e-3, "f64": 1e-12, "dd": 1e-25}[prec]


@pytest.mark.gpu
def test_jax_advance_matches_numpy():
    jax = pytest.importorskip("jax")
    case = load_case("nb3_pilot")
    k_np = BatchedTaylor(case.tape, 16, "f64", "numpy")
    k_jx = BatchedTaylor(case.tape, 16, "f64", "jax")
    z_np, t_np, s_np = k_np.advance(k_np.from_fractions([case.ic(0)]), np.zeros(1), 4.0)
    z_jx, t_jx, s_jx = k_jx.advance(k_jx.from_fractions([case.ic(0)]), jax.numpy.zeros(1), 4.0)
    assert float(t_jx[0]) == 4.0
    assert int(s_jx[0]) == int(s_np[0])
    assert np.abs(np.asarray(z_jx) - z_np).max() < 1e-11
