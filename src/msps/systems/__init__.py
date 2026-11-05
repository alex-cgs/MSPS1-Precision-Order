"""Case cards: YAML in cases/ plus the builders here (spec §6). load_case('lorenz63:rho_28') resolves a variant."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import yaml

from ..symbolic import build_tape, compile_coeffs, compile_values
from ..symbolic.poly import PolySystem
from . import ic as _ic
from . import nbody as _nbody
from . import odes as _odes

ROOT = Path(os.environ.get("MSPS_ROOT", Path(__file__).resolve().parents[3]))
_cache: dict = {}


_SECTIONS = ("T", "ic", "reference", "tolerances", "regime", "oracle")


def _params(card, variant):
    """Variant keys override params; the sections in _SECTIONS override the card itself."""
    p = dict(card.get("params", {}))
    if variant is not None:
        over = dict(card["variants"][variant] or {})
        for key in _SECTIONS:
            if key in over:
                val = over.pop(key)
                card[key] = {**card[key], **val} if isinstance(val, dict) else val
        p.update(over)
    return p


def _build(name, card, params) -> PolySystem:
    builder = card["builder"]
    if builder == "nbody":
        N = int(params.get("N", len(params.get("masses", []))))
        ms = params.get("masses")
        if ms is None:
            pattern = params.get("mass_pattern", ["1"])
            ms = [pattern[i % len(pattern)] for i in range(N)]
        if card["ic"].get("generator") == "kepler":
            e = _ic.parse(params["e"])
            ms = [str((1 - e * e) / 2)] * 2
        return _nbody.build(name, [_ic.parse(m) for m in ms], int(params.get("dim", 2)))
    return getattr(_odes, builder)(name, {k: (_ic.parse(v) if k != "N" else int(v)) for k, v in params.items()})


@dataclass
class Case:
    name: str
    variant: str | None
    card: dict
    params: dict
    system: PolySystem
    _attempt: int = 0
    _tape: object = None
    _coeffs: object = None
    _value_fns: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name if self.variant is None else f"{self.name}:{self.variant}"

    @property
    def tape(self):
        if self._tape is None:
            self._tape = build_tape(self.system)
        return self._tape

    @property
    def coeffs_fn(self):
        if self._coeffs is None:
            self._coeffs = compile_coeffs(self.tape)
        return self._coeffs

    def value_fn(self, key: str):
        """Compiled (fn, constants) for 'rhs', 'invariants' or 'constraints'."""
        if key not in self._value_fns:
            S = self.system
            exprs = {"rhs": S.rhs_list(), "invariants": list(S.invariants.values()),
                     "constraints": list(S.constraints.values())}[key]
            self._value_fns[key] = compile_values(S.states, exprs, key)
        return self._value_fns[key]

    def ic(self, ic_id: int = 0) -> list:
        """Exact rational initial state, full lifted vector. Cached on disk: generators may run filter integrations."""
        path = ROOT / "results" / "references" / "ics" / f"{self.id.replace(':', '__')}_{ic_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return [Fraction(x) for x in data["state"]]
        z, info = self._generate(ic_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"case": self.id, "ic_id": ic_id, "state": [str(x) for x in z],
                                   "names": [str(s) for s in self.system.states], "filter": info}))
        tmp.replace(path)
        return z

    def _generate(self, ic_id):
        gen = _ic.GENERATORS[self.card["ic"]["generator"]]
        flt = self.card["ic"].get("filter")
        info = {"attempts": 1}
        rejects = {}
        for attempt in range(100000):
            self._attempt = attempt
            base = gen(self, ic_id)
            try:
                z = self.system.lift(base)
            except ValueError:  # coincident bodies from the generator: draw again
                continue
            if not flt:
                return z, info
            ok, why = _passes_filter(self, z, flt)
            if ok:
                return z, {"attempts": attempt + 1, "rejects": rejects, **why}
            rejects[why["reject"]] = rejects.get(why["reject"], 0) + 1
        raise RuntimeError(f"no IC passed the filter for {self.id} #{ic_id}")


def _passes_filter(case, z, flt):
    """Screening on an f64 orbit. A chaotic run leaves the true orbit early, so this bounds its cost only, not the reference's."""
    import gmpy2

    from ..arith import get_arith
    prec = gmpy2.get_context().precision
    try:
        arith = get_arith("f64")
        arith.activate()
        K = [arith.convert(c) for c in case.value_fn("invariants")[1]]
        energy = case.value_fn("invariants")[0](z, K)[0]
        if "energy_max" in flt and energy >= _ic.parse(flt["energy_max"]):
            return False, {"reject": "energy"}
        if _pair_extent(case.system, z, min) < float(_ic.parse(flt["dmin0"])):
            return False, {"reject": "dmin0"}
        ok, why = _filter_orbit(case, z, arith, int(flt["order"]), int(flt["max_steps"]), flt)
        if not ok:
            return False, why
        return True, {"energy": float(energy), "filter_steps": why["steps"]}
    finally:
        gmpy2.get_context().precision = prec


def _filter_orbit(case, z, arith, order, max_steps, flt):
    from ..integrators.taylor import TaylorStepper
    from ..schedules import AdaptiveDyadic
    st = TaylorStepper(case, order, arith)
    sched = AdaptiveDyadic(safety=8, cap=Fraction(1, 2))
    Z = [arith.convert(v) for v in z]
    t, T, steps = Fraction(0), _ic.parse(flt["T"]), 0
    dmax = float(_ic.parse(flt["dmax"]))
    while t < T:
        c = st.coeffs(Z)
        h = min(sched.next_h(t, Z, c, st), T - t)
        Z = st.horner(c, arith.convert(h))
        t += h
        steps += 1
        if steps > max_steps:
            return False, {"reject": "max_steps"}
        if steps % 8 == 0 and _pair_extent(case.system, Z, max) > dmax:
            return False, {"reject": "escape", "t": float(t)}
    return True, {"steps": steps}


def _pair_extent(S, z, fn):
    N, dim = S.meta["N"], S.meta["dim"]
    ds = []
    for i, j in S.meta["pairs"]:
        ds.append(sum((float(z[a * N + i]) - float(z[a * N + j])) ** 2 for a in range(dim)) ** 0.5)
    return fn(ds)


def load_card(name: str) -> dict:
    return yaml.safe_load((ROOT / "cases" / f"{name}.yaml").read_text())


def load_case(spec: str) -> Case:
    if spec in _cache:
        return _cache[spec]
    name, _, variant = spec.partition(":")
    card = load_card(name)
    variant = variant or card.get("default_variant")
    if card.get("kind") == "map":
        raise ValueError(f"{name} is a map; use msps.integrators.maps")
    params = _params(card, variant)
    case = Case(name, variant, card, params, _build(name, card, params))
    _cache[spec] = case
    return case


def all_cards() -> list:
    return sorted(p.stem for p in (ROOT / "cases").glob("*.yaml"))
