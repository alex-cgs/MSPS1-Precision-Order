"""Iterated maps over any arithmetic (spec §6.2): logistic, tent, sine and its Taylor truncations T_w."""
from __future__ import annotations

from fractions import Fraction

import sympy as sp

from ..symbolic import compile_values
from ..systems import load_card
from ..systems.ic import parse


def resolve_card(spec: str) -> tuple:
    name, _, variant = spec.partition(":")
    card = load_card(name)
    params = dict(card.get("params", {}))
    if variant:
        params.update(card["variants"][variant] or {})
    return card, params


class IteratedMap:
    def __init__(self, spec: str, arith, overrides: dict | None = None):
        self.card, self.params = resolve_card(spec)
        self.params.update(overrides or {})
        self.arith = arith
        kind = self.card["builder"]
        x = sp.Symbol("x", real=True)
        p = self.params
        self.degree = 1
        self._branch = None
        if kind == "logistic":
            expr = sp.Rational(str(parse(p["r"]))) * x * (1 - x)
            self.degree = 2
        elif kind == "tent":
            self.mu = arith.convert(parse(p["mu"]))
            self.half = arith.convert(Fraction(1, 2))
            self.one = arith.convert(Fraction(1))
            self._branch = "tent"
            expr = None
        elif kind == "sine":
            mu = parse(p["mu"])
            pi_q = parse(p.get("pi", "pi"))
            w = p.get("w")
            if w is None:
                self._branch = "sine"
                self.mu = arith.convert(mu)
                self.pi = arith.convert(pi_q)
                expr = None
            else:
                w = int(w)
                pr = sp.Rational(pi_q.numerator, pi_q.denominator)
                # T_w: Taylor polynomial of sin of degree 2w+1, Horner-factored in (pi x)^2 for the tape compiler
                y = pr * x
                acc = sp.Rational((-1) ** w, sp.factorial(2 * w + 1))
                for j in range(w - 1, -1, -1):
                    acc = acc * y**2 + sp.Rational((-1) ** j, sp.factorial(2 * j + 1))
                expr = sp.Rational(mu.numerator, mu.denominator) * y * acc
                self.degree = 2 * w + 1
        else:
            raise ValueError(kind)
        if expr is not None:
            self._fn, c = compile_values([x], [expr], "mapfn")
            self.K = [arith.convert(v) for v in c]

    def step(self, x):
        if self._branch == "tent":
            return self.mu * x if x < self.half else self.mu * (self.one - x)
        if self._branch == "sine":
            return self.mu * self.arith.sin(self.pi * x)
        return self._fn([x], self.K)[0]

    def iterate(self, x, count: int):
        for _ in range(count):
            x = self.step(x)
        return x
