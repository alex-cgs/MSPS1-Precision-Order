"""Polynomial ODE systems and SymPy polynomialization (spec §4.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sp

IC_DIGITS = 310  # digits for auxiliary initial data that are not rational (about 1030 bits; beyond every p used)


@dataclass
class PolySystem:
    name: str
    states: list  # sympy Symbols, state order
    rhs: dict  # Symbol -> polynomial expression (left unexpanded so the tape can share subexpressions)
    invariants: dict = field(default_factory=dict)  # name -> polynomial expression in the states
    constraints: dict = field(default_factory=dict)  # name -> polynomial expression, 0 on the true manifold
    positions: list = field(default_factory=list)  # state indices entering E(t) (spec §4.7)
    aux_init: dict = field(default_factory=dict)  # auxiliary Symbol -> expression in base states
    base: list = field(default_factory=list)  # states that initial conditions specify directly
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.states)

    @property
    def degree(self) -> int:
        return max(sp.Poly(sp.expand(e), *self.states).total_degree() for e in self.rhs.values())

    def rhs_list(self) -> list:
        return [self.rhs[s] for s in self.states]

    def lift(self, base_values: dict) -> list:
        """Full rational initial state from base values; auxiliaries exact when rational, else IC_DIGITS digits."""
        point = {k: sp.Rational(v.numerator, v.denominator) for k, v in base_values.items()}
        out = []
        for s in self.states:
            if s in point:
                out.append(Fraction(int(point[s].p), int(point[s].q)))
            else:
                out.append(_to_fraction(self.aux_init[s].subs(point)))
        return out


def _to_fraction(val) -> Fraction:
    if val.is_Rational:
        return Fraction(int(val.p), int(val.q))
    r = sp.Rational(str(sp.N(val, IC_DIGITS)))
    return Fraction(int(r.p), int(r.q))


def lie(expr, system: PolySystem):
    """Lie derivative L_f expr, polynomial in the states."""
    return sum(sp.diff(expr, s) * system.rhs[s] for s in system.states)


def polynomialize(name: str, states: list, rhs: dict, positions=None, invariants=None, meta=None) -> PolySystem:
    """Rewrite `rhs` into polynomial form by adding auxiliaries (spec §4.1).

    sin(g), cos(g) -> (s, c);  exp(g) -> e;  g^(-1) -> w with w' = -w^2 g';  g^(-1/2) -> v with v' = -v^3 g'/2.
    Each auxiliary records a polynomial constraint that holds exactly on the true solution.
    """
    states = list(states)
    base = list(states)
    rhs = dict(rhs)
    inv = dict(invariants or {})
    aux_init, constraints = {}, {}
    counter = [0]

    def fresh(prefix):
        counter[0] += 1
        return sp.Symbol(f"{prefix}{counter[0]}", real=True)

    def ddt(g):
        return sum(sp.diff(g, s) * rhs[s] for s in states if g.has(s))

    def find(exprs):
        for e in exprs:
            for a in sp.postorder_traversal(e):  # innermost first, so g in g^-1 is polynomial when reached
                if isinstance(a, (sp.sin, sp.cos, sp.exp)):
                    return a
                if isinstance(a, sp.Pow) and a.has(*states):
                    ex = a.exp
                    if ex.is_Integer and ex > 0:
                        continue
                    if ex.is_Integer or (ex.is_Rational and ex.q == 2 and ex < 0):
                        return a
                    raise ValueError(f"cannot polynomialize {a}")
        return None

    while True:
        atom = find(list(rhs.values()) + list(inv.values()))
        if atom is None:
            break
        if isinstance(atom, (sp.sin, sp.cos)):
            g = atom.args[0]
            s, c = fresh("s"), fresh("c")
            dg = ddt(g)
            aux_init[s], aux_init[c] = _base_expr(sp.sin(g), aux_init), _base_expr(sp.cos(g), aux_init)
            rhs[s], rhs[c] = c * dg, -s * dg
            states += [s, c]
            constraints[f"unit_{s}_{c}"] = s**2 + c**2 - 1
            sub = {sp.sin(g): s, sp.cos(g): c}
        elif isinstance(atom, sp.exp):
            g = atom.args[0]
            e = fresh("e")
            aux_init[e] = _base_expr(sp.exp(g), aux_init)
            rhs[e] = e * ddt(g)
            states.append(e)
            sub = {sp.exp(g): e}
        else:
            g, ex = atom.base, atom.exp
            if ex.is_Integer:
                w = fresh("w")
                aux_init[w] = _base_expr(1 / g, aux_init)
                rhs[w] = -w**2 * ddt(g)
                states.append(w)
                constraints[f"inv_{w}"] = w * g - 1
                sub = {g ** k: w ** (-k) for k in _powers(list(rhs.values()) + list(inv.values()), g, integer=True)}
            else:
                v = fresh("v")
                aux_init[v] = _base_expr(g ** sp.Rational(-1, 2), aux_init)
                rhs[v] = -sp.Rational(1, 2) * v**3 * ddt(g)
                states.append(v)
                constraints[f"invsqrt_{v}"] = v**2 * g - 1
                sub = {g ** k: v ** int(-2 * k) for k in _powers(list(rhs.values()) + list(inv.values()), g, integer=False)}
        rhs = {k: e.xreplace(sub) for k, e in rhs.items()}
        inv = {k: e.xreplace(sub) for k, e in inv.items()}
    return PolySystem(name, states, rhs, invariants=inv, constraints=constraints,
                      positions=list(positions if positions is not None else range(len(base))),
                      aux_init=aux_init, base=base, meta=meta or {})


def _powers(exprs, g, integer):
    found = set()
    for e in exprs:
        for a in sp.preorder_traversal(e):
            if isinstance(a, sp.Pow) and a.base == g and a.exp < 0:
                if integer == bool(a.exp.is_Integer):
                    found.add(a.exp)
    return found


def _base_expr(expr, aux_init):
    """Express an auxiliary's definition through base states only (arguments may contain earlier auxiliaries)."""
    return expr.xreplace(aux_init) if aux_init else expr


def check_invariant(system: PolySystem, expr) -> bool:
    """True when L_f expr vanishes identically, no constraint needed (spec §4.1)."""
    return sp.expand(lie(expr, system)) == 0


def check_constraint_closed(system: PolySystem, expr) -> bool:
    """True when L_f C = g * C for polynomial g, so C = 0 is invariant (errors neither damped nor corrected)."""
    q, r = sp.div(sp.Poly(sp.expand(lie(expr, system)), *system.states), sp.Poly(sp.expand(expr), *system.states))
    return r.is_zero
