import importlib.util
import math
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import gmpy2
import numpy as np
import pytest
import sympy as sp

from msps.arith import get_arith
from msps.integrators import DOP853, RK4, GaussLegendre, SplitStepper, TaylorStepper
from msps.integrators.maps import IteratedMap
from msps.runner.window import Window
from msps.symbolic import check_constraint_closed, check_invariant, lie_coefficients, polynomialize
from msps.systems import load_case
from msps.systems import nbody as nb

ROOT = Path(__file__).resolve().parents[1]


def _pilot():
    spec = importlib.util.spec_from_file_location("taylor3b", ROOT / "pilot" / "taylor3b.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tape_exact(case, z0, order):
    q = get_arith("q")
    st = TaylorStepper(case, order, q)
    c = st.coeffs([q.convert(v) for v in z0])
    return [[q.to_fraction(x) for x in ci] for ci in c]


LIE_CASES = ["nb3_pilot", "lorenz63", "henon_heiles:E_1_8"]


@pytest.mark.parametrize("spec", LIE_CASES)
def test_tape_equals_lie_series(spec):
    case = load_case(spec)
    z0 = _rational_point(case)
    lie = lie_coefficients(case.system, z0, 4)
    tape = _tape_exact(case, z0, 4)
    for i in lie:
        assert lie[i] == tape[i], case.system.states[i]


def _rational_point(case):
    rng = np.random.default_rng(1)
    return [Fraction(int(rng.integers(1, 40)), int(rng.integers(1, 12))) for _ in case.system.states]


@pytest.mark.parametrize("N", [2, 3])
def test_nbody_tape_lie_any_N(N):
    S = nb.build("t", [Fraction(k + 1, 2) for k in range(N)])
    case = _bare_case(S)
    z0 = _rational_point(case)
    lie = lie_coefficients(S, z0, 4)
    tape = _tape_exact(case, z0, 4)
    for i in lie:
        assert lie[i] == tape[i]


@pytest.mark.slow
def test_nbody_tape_lie_N4():
    S = nb.build("t", [Fraction(1)] * 4)
    case = _bare_case(S)
    z0 = _rational_point(case)
    # all position and velocity components of body 0 plus one pair variable, orders 0..4
    vars_ = [0, 4, 8, 12, S.states.index(sp.Symbol("u0_1", real=True))]
    lie = lie_coefficients(S, z0, 4, variables=vars_)
    tape = _tape_exact(case, z0, 4)
    for i in lie:
        assert lie[i] == tape[i]


def _bare_case(S):
    from msps.systems import Case
    return Case(S.name, None, {"ic": {}}, {}, S)


def test_tape_reproduces_pilot_exactly():
    pilot = _pilot()
    import flint
    case = load_case("nb3_pilot")
    z0 = case.ic(0)
    assert z0 == pilot.initial_state()
    c_pilot = pilot.taylor_coeffs([flint.fmpq(v.numerator, v.denominator) for v in z0], 12)
    tape = _tape_exact(case, z0, 12)
    for ci, ti in zip(c_pilot, tape):
        assert [Fraction(int(x.p), int(x.q)) for x in ci] == ti


def test_generic_polynomialize_matches_gravity_form():
    x, y, vx, vy = sp.symbols("x y vx vy", real=True)
    r2 = x**2 + y**2
    S = polynomialize("kepler", [x, y, vx, vy], {x: vx, y: vy, vx: -x * r2 ** sp.Rational(-3, 2), vy: -y * r2 ** sp.Rational(-3, 2)})
    v = S.states[4]
    assert sp.expand(S.rhs[v] + v**3 * (x * vx + y * vy)) == 0
    assert sp.expand(S.rhs[vx] + x * v**3) == 0
    assert check_constraint_closed(S, list(S.constraints.values())[0])


def test_gravity_invariants_and_constraint_symbolic():
    S = load_case("nb3_pilot").system
    for k in S.invariants:
        assert check_invariant(S, S.invariants[k]), k
    assert check_constraint_closed(S, S.constraints["C0_1"])


def test_lifted_systems_build():
    for spec in ["double_pendulum", "duffing", "fput_beta:N_16", "lorenz96:N_8", "rossler", "nbN_circle:N_4", "nbN_alternating:hier_N_4", "nbN_cluster:N_4"]:
        case = load_case(spec)
        assert case.tape.dim == case.system.dim
        assert case.system.degree >= 2


def test_lifted_ic_and_invariants_double_pendulum():
    case = load_case("double_pendulum")
    z = case.ic(0)
    for name, expr in case.system.constraints.items():
        v = expr.subs({s: sp.Rational(a.numerator, a.denominator) for s, a in zip(case.system.states, z)})
        assert abs(v) < sp.Rational(1, 10**250), name
    assert check_invariant_numeric(case)


def check_invariant_numeric(case):
    a = get_arith("mpfr-300")
    a.activate()
    st = TaylorStepper(case, 30, a)
    z = [a.convert(v) for v in case.ic(0)]
    fi, ki = case.value_fn("invariants")
    K = [a.convert(c) for c in ki]
    H0 = fi(z, K)[0]
    for _ in range(4):
        z = st.step(z, a.convert(Fraction(1, 64)))
    return abs(float(fi(z, K)[0] - H0)) < 1e-30


@pytest.mark.parametrize("spec,n", [("lorenz63", 4), ("lorenz63", 8), ("nb3_pilot", 6), ("henon_heiles:E_1_8", 5)])
def test_one_step_convergence_order(spec, n):
    case = load_case(spec)
    a = get_arith("mpfr-600")
    a.activate()
    st = TaylorStepper(case, n, a)
    hi = TaylorStepper(case, n + 40, a)
    z = [a.convert(v) for v in case.ic(0)]
    hs = [Fraction(1, 2**k) for k in range(5, 10)]
    errs = []
    for h in hs:
        hc = a.convert(h)
        e = max(abs(x - y) for x, y in zip(st.step(z, hc), hi.step(z, hc)))
        errs.append(math.log2(float(e)))
    slope = np.polyfit([math.log2(h) for h in hs], errs, 1)[0]
    assert abs(slope - (n + 1)) < 0.25


@pytest.mark.parametrize("cls,args,order", [(RK4, (), 4), (DOP853, (), 8), (GaussLegendre, (2, 1e-40), 4), (GaussLegendre, (3, 1e-40), 6)])
def test_rk_orders(cls, args, order):
    case = load_case("lorenz63")
    a = get_arith("mpfr-300" if cls is not DOP853 else "dd")
    a.activate()
    st = cls(case, a, *args)
    hi = TaylorStepper(case, 40, a)
    z = [a.convert(v) for v in case.ic(0)]
    hs = [Fraction(1, 2**k) for k in range(8, 12)] if cls is not DOP853 else [Fraction(1, 2**k) for k in range(5, 9)]
    errs = []
    for h in hs:
        hc = a.convert(h)
        e = max(abs(float(a.to_mpfr(x) - a.to_mpfr(y))) for x, y in zip(st.step(z, hc), hi.step(z, hc)))
        errs.append(math.log2(e))
    slope = np.polyfit([math.log2(h) for h in hs], errs, 1)[0]
    assert abs(slope - (order + 1)) < 0.7


@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_yoshida_order(k):
    case = load_case("henon_heiles:E_1_8")
    a = get_arith("dd")
    a.activate()
    st = SplitStepper(case, a, k)
    hi = TaylorStepper(case, 30, a)
    z = [a.convert(v) for v in case.ic(0)]
    hs = [Fraction(1, 2**j) for j in (4, 5, 6)] if k < 8 else [Fraction(1, 2**j) for j in (2, 3, 4)]
    errs = []
    for h in hs:
        hc = a.convert(h)
        e = max(abs(float(a.to_mpfr(x) - a.to_mpfr(y))) for x, y in zip(st.step(z, hc), hi.step(z, hc)))
        errs.append(math.log2(e))
    slope = np.polyfit([math.log2(h) for h in hs], errs, 1)[0]
    assert slope > k + 0.3


def kepler_position(e, t, dps=120):
    import mpmath as mp
    with mp.workdps(dps):
        e, t = mp.mpf(e.numerator) / e.denominator, mp.mpf(t.numerator) / t.denominator
        n = mp.sqrt(1 - e**2)
        M = n * t
        E = mp.findroot(lambda E_: E_ - e * mp.sin(E_) - M, M + e / 2)
        rx, ry = mp.cos(E) - e, mp.sqrt(1 - e**2) * mp.sin(E)
        return float(rx), float(ry), mp.nstr(rx, 80), mp.nstr(ry, 80)


@pytest.mark.parametrize("variant,e", [("e_0", Fraction(0)), ("e_1_2", Fraction(1, 2))])
def test_kepler_oracle(variant, e):
    case = load_case(f"nb2_kepler:{variant}")
    a = get_arith("mpfr-300")
    a.activate()
    st = TaylorStepper(case, 30, a)
    z = [a.convert(v) for v in case.ic(0)]
    h = Fraction(1, 16)
    steps = 160
    for _ in range(steps):
        z = st.step(z, a.convert(h))
    _, _, xs, ys = kepler_position(e, h * steps)
    # relative separation q2 - q1 at the periapsis-aligned frame; bodies start with the relative vector along +x
    rx = z[1] - z[0]
    ry = z[3] - z[2]
    assert abs(float(rx - gmpy2.mpfr(xs))) < 1e-25
    assert abs(float(ry - gmpy2.mpfr(ys))) < 1e-25


@pytest.mark.parametrize("spec,arith", [("nb3_pilot", "f64"), ("nb3_pilot", "dd"), ("nb3_pilot", "mpfr-113"), ("henon_heiles:E_1_8", "f64")])
def test_invariants_at_rounding_level(spec, arith):
    case = load_case(spec)
    a = get_arith(arith)
    a.activate()
    st = TaylorStepper(case, 24, a)
    z = [a.convert(v) for v in case.ic(0)]
    from msps.diagnostics import InvariantMonitor
    mon = InvariantMonitor(case, a, z)
    h = a.convert(Fraction(1, 32))
    for _ in range(64):
        z = st.step(z, h)
    d = mon.deltas(z)
    u = a.unit_roundoff()
    for k, v in d.items():
        assert v < 3e4 * u, (k, v)


def test_backends_agree_at_high_precision():
    case = load_case("lorenz63")
    z0 = case.ic(0)
    h = Fraction(1, 64)
    results = {}
    for name in ["mpfr-1000", "q", "qr-1000", "arb-1000"]:
        a = get_arith(name)
        a.activate()
        st = TaylorStepper(case, 16, a)
        z = [a.convert(v) for v in z0]
        for _ in range(4):
            z = st.step(z, a.convert(h))
        results[name] = [a.to_fraction(v) for v in z]
    base = results["q"]
    for name, vals in results.items():
        assert max(abs(float(x - y)) for x, y in zip(vals, base)) < 2.0 ** -900, name


@pytest.mark.parametrize("name,bits", [("f32", 24), ("f64", 53), ("f80", 64), ("dd", 106), ("mpfr-80", 80)])
def test_lower_precision_backends_agree_to_rounding(name, bits):
    case = load_case("lorenz63")
    z0 = case.ic(0)
    h = Fraction(1, 64)
    ref = get_arith("mpfr-600")
    ref.activate()
    st = TaylorStepper(case, 12, ref)
    r = [ref.convert(v) for v in z0]
    for _ in range(2):
        r = st.step(r, ref.convert(h))
    R = [ref.to_fraction(v) for v in r]
    a = get_arith(name)
    a.activate()
    st = TaylorStepper(case, 12, a)
    z = [a.convert(v) for v in z0]
    for _ in range(2):
        z = st.step(z, a.convert(h))
    Z = [a.to_fraction(v) for v in z]
    scale = max(abs(float(v)) for v in R)
    assert max(abs(float(x - y)) for x, y in zip(Z, R)) < scale * 2.0 ** (-bits + 7)


def test_dd_basics_exact():
    a = get_arith("dd")
    x = a.convert(Fraction(1, 3))
    assert abs(a.to_fraction(x * a.convert(Fraction(3))) - 1) < Fraction(1, 2**100)
    y = x / 7
    assert abs(a.to_fraction(y) - Fraction(1, 21)) < Fraction(1, 2**105)
    s = a.convert(Fraction(2)).sqrt()
    assert abs(a.to_fraction(s * s) - 2) < Fraction(1, 2**100)


def test_qr_rounding_denominator():
    a = get_arith("qr-53")
    v = a.post_step([a.convert(Fraction(1, 3))])[0]
    assert int(v.q) <= 2**53
    assert abs(a.to_fraction(v) - Fraction(1, 3)) <= Fraction(1, 2**54)


def test_maps_logistic_tent_collapse():
    a = get_arith("f64")
    a.activate()
    m = IteratedMap("tent_map:mu_2", a)
    assert m.iterate(a.convert(Fraction(123456789, 2**30) * 1), 70) == 0.0
    q = get_arith("q")
    q.activate()
    lm = IteratedMap("logistic_map", q)
    x = q.convert(Fraction(1, 3))
    y = lm.iterate(x, 2)
    assert q.to_fraction(y) == Fraction(32, 81)


@pytest.mark.parametrize("hm,expected", [((2, 29), False), ((2, 30), True), ((3, 0), True), ((4, 24), True), ((4, 25), False), ((5, 0), False)])
def test_window_may_start(hm, expected):
    w = Window(clock=lambda: datetime(2026, 11, 5, hm[0], hm[1]))
    assert w.may_start(30) is expected


def test_window_refuses_jobs_too_long():
    w = Window(clock=lambda: datetime(2026, 11, 5, 4, 0))
    assert w.seconds_left() == 25 * 60
    assert w.may_start(24 * 60)
    assert not w.may_start(26 * 60)
    assert w.may_start(26 * 60, checkpoint_s=120)
    assert not w.may_start(26 * 60, checkpoint_s=26 * 60)
    w2 = Window(clock=lambda: datetime(2026, 11, 5, 4, 26))
    assert w2.must_stop()
    assert not w2.may_start(1)


def test_window_clock_advances_through_a_campaign():
    t = [datetime(2026, 11, 5, 4, 24, 0)]
    w = Window(clock=lambda: t[0])
    assert not w.must_stop(margin_s=30)
    t[0] = datetime(2026, 11, 5, 4, 24, 40)
    assert w.must_stop(margin_s=30)
    assert w.next_start() == datetime(2026, 11, 6, 2, 30)


def test_filter_truth_stage_rejects_and_restores_precision():
    import gmpy2

    from msps.systems import _passes_filter
    case = load_case("nb3_pilot")
    z = case.ic(0)
    flt = {"dmin0": "0", "dmax": "100", "T": "1", "order": 8, "max_steps": 10000, "truth": {"arith": "mpfr-64", "order": 8}}
    gmpy2.get_context().precision = 777
    ok, why = _passes_filter(case, z, flt)
    assert ok and why["truth_steps"] > 0
    gmpy2.get_context().precision = 777
    ok, why = _passes_filter(case, z, {**flt, "truth": {"arith": "mpfr-64", "order": 8, "max_steps": 1}})
    assert not ok and why["reject"] == "truth_max_steps" and gmpy2.get_context().precision == 777


def test_reference_step_budget_and_failed_check(monkeypatch, tmp_path):
    import pytest

    from msps.runner import references as R
    case = load_case("nb3_pilot")
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    monkeypatch.setitem(case.card, "T", 1)
    monkeypatch.setitem(case.card["reference"], "max_steps", 2)
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        R.compute_reference(case.id, 0)
    monkeypatch.delitem(case.card["reference"], "max_steps")
    monkeypatch.setitem(case.card["reference"], "tol_check", 0.0)
    assert not R.compute_reference(case.id, 0)["check_passed"]
    with pytest.raises(RuntimeError, match="failed its check"):
        R.Reference(case, 0)


def test_queue_counts_and_pending_follow_the_current_grid(monkeypatch, tmp_path):
    from msps import io as rio
    from msps.runner.queue import Queue, run_id
    monkeypatch.setattr(rio, "RESULTS", tmp_path)
    old, new = {"kind": "traj", "case": "old", "ic": 0}, {"kind": "traj", "case": "new", "ic": 0}
    q = Queue("T")
    q.add([("g", old), ("g", new)])
    assert q.counts({run_id("T", new)}) == {"pending": 1} and q.counts() == {"pending": 2}


def test_superseded_cases_are_hidden_but_kept(monkeypatch, tmp_path):
    import json

    from msps import io as rio
    root = tmp_path / "results"
    monkeypatch.setattr(rio, "RESULTS", root)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "superseded.yaml").write_text("cases: {old: why}\n")
    for name, case in (("a", "old"), ("b", "new")):
        d = root / "raw" / "T" / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"config": {"case": case}}))
    assert [m["config"]["case"] for _, m in rio.iter_runs("T")] == ["new"]
    assert len(list(rio.iter_runs("T", include_superseded=True))) == 2


def test_radius_survives_coefficients_beyond_float_range():
    case = load_case("nb3_pilot")
    a = get_arith("mpfr-200")
    a.activate()
    st = TaylorStepper(case, 4, a)
    big = gmpy2.mpfr("1e400")
    one = gmpy2.mpfr(1)
    assert st.radius([[one, one, one, one, big]]) == pytest.approx(10.0**-100, rel=1e-9)
    f = get_arith("f64")
    f.activate()
    st = TaylorStepper(case, 4, f)
    assert st.radius([[1.0, 1.0, 1.0, 1.0, float("inf")]]) == 0.0
    assert st.radius([[1.0, 1.0, 1.0, 16.0, 16.0]]) == pytest.approx(min(16.0 ** (-1 / 3), 16.0 ** (-1 / 4)))
