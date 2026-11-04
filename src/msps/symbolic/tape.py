"""Taylor-tape generator (spec §4.2): polynomial RHS -> straight-line add/sub/mul/scale/const program.

Node series x_[0..n]; state nodes obey x_[k+1] = f_[k]/(k+1); products are Cauchy products. Hash-consing with
sign normalization shares subexpressions, so (qx_j - qx_i) and (qx_i - qx_j) use one node. The tape is compiled
to Python source. `functools.reduce` replaces `sum` on purpose: since Python 3.12 `sum` uses compensated
summation for floats, which would silently raise the effective precision of the f64 path.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce  # noqa: F401  (used by generated code)
from operator import add, mul  # noqa: F401

import sympy as sp

from .poly import PolySystem


@dataclass
class TaylorTape:
    system_name: str
    dim: int
    ops: list  # (kind, dst, a, b): add sub mul neg; scale (b = const idx); const (a = const idx); isqrt
    rhs: list  # state index -> node id
    constants: list  # Fractions
    n_nodes: int

    @property
    def n_mul(self) -> int:
        return sum(1 for o in self.ops if o[0] == "mul")

    def counts(self) -> dict:
        out = {}
        for o in self.ops:
            out[o[0]] = out.get(o[0], 0) + 1
        return out


class _Builder:
    def __init__(self, states):
        self.index = {s: i for i, s in enumerate(states)}
        self.n_states = len(states)
        self.next = len(states)
        self.ops, self.cache, self.constants, self.cidx = [], {}, [], {}

    def _const_id(self, q: Fraction) -> int:
        if q not in self.cidx:
            self.cidx[q] = len(self.constants)
            self.constants.append(q)
        return self.cidx[q]

    def _new(self, key):
        if key in self.cache:
            return self.cache[key]
        nid = self.next
        self.next += 1
        self.ops.append((key[0], nid, key[1], key[2] if len(key) > 2 else None))
        self.cache[key] = nid
        return nid

    def const(self, q: Fraction) -> int:
        return self._new(("const", self._const_id(q)))

    def mul_nodes(self, a, b):
        return self._new(("mul", min(a, b), max(a, b)))

    def power(self, a, k: int):
        node = a
        for _ in range(k - 1):
            node = self.mul_nodes(node, a) if node != a else self._new(("mul", a, a))
        return node

    def materialize(self, coef: Fraction, node) -> int:
        if coef == 1:
            return node
        if coef == -1:
            return self._new(("neg", node))
        return self._new(("scale", node, self._const_id(coef)))

    def term(self, expr):
        """(coef, node) with coef * node == expr; node None means the pure constant coef."""
        if expr.is_Symbol:
            return Fraction(1), self.index[expr]
        if expr.is_Rational:
            return Fraction(int(expr.p), int(expr.q)), None
        if expr.is_Add:
            return self._add(expr)
        if expr.is_Mul:
            coef = Fraction(1)
            nodes = []
            for arg in expr.args:
                c, nd = self.term(arg)
                coef *= c
                if nd is not None:
                    nodes.append(nd)
            if coef == 0:
                return Fraction(0), None
            if not nodes:
                return coef, None
            nodes.sort(key=lambda n: (n >= self.n_states, n))
            acc = nodes[-1]
            for nd in reversed(nodes[:-1]):
                acc = self.mul_nodes(nd, acc)
            return coef, acc
        if expr.is_Pow:
            base, ex = expr.base, expr.exp
            if ex.is_Integer and ex > 0:
                c, nd = self.term(base)
                return c ** int(ex), self.power(nd, int(ex)) if nd is not None else None
            if ex.is_Rational and ex.q == 2 and ex < 0:
                c, nd = self.term(base)
                base_node = self.materialize(c, nd if nd is not None else self.const(Fraction(1)))
                inv = self._new(("isqrt", base_node))
                return Fraction(1), self.power(inv, int(-ex.p))
        raise ValueError(f"not polynomial: {expr}")

    def _add(self, expr):
        terms = {}
        constant = Fraction(0)
        for arg in expr.args:
            c, nd = self.term(arg)
            if nd is None:
                constant += c
            else:
                terms[nd] = terms.get(nd, Fraction(0)) + c
        items = sorted((n, c) for n, c in terms.items() if c != 0)
        if not items:
            return constant, None
        lead = items[0][1]
        items = [(n, c / lead) for n, c in items]
        if constant != 0:
            items.append((self.const(constant / lead), Fraction(1)))
        key = ("lin",) + tuple(items)
        if key in self.cache:
            return lead, self.cache[key]
        acc = items[0][0]
        for n, c in items[1:]:
            if c == 1:
                acc = self._new(("add", acc, n))
            elif c == -1:
                acc = self._new(("sub", acc, n))
            else:
                acc = self._new(("add", acc, self.materialize(c, n)))
        self.cache[key] = acc
        return lead, acc

    def node(self, expr) -> int:
        c, nd = self.term(expr)
        if nd is None:
            return self.const(c)
        return self.materialize(c, nd)


def build_tape(system: PolySystem) -> TaylorTape:
    b = _Builder(system.states)
    rhs = [b.node(sp.sympify(system.rhs[s])) for s in system.states]
    return TaylorTape(system.name, system.dim, b.ops, rhs, b.constants, b.next)


def compile_coeffs(tape: TaylorTape):
    """f(z, n, K, ZERO) -> list over states of coefficient lists [x_[0], ..., x_[n]]. K: converted constants."""
    D = tape.dim
    head = [f"    x{i} = [z[{i}]]" for i in range(D)]
    body = []
    for kind, dst, a, b in tape.ops:
        if kind == "const":
            head.append(f"    x{dst} = [K[{a}]] + [ZERO] * n")
        else:
            head.append(f"    x{dst} = []")
    for kind, dst, a, b in tape.ops:
        if kind == "add":
            body.append(f"x{dst}.append(x{a}[k] + x{b}[k])")
        elif kind == "sub":
            body.append(f"x{dst}.append(x{a}[k] - x{b}[k])")
        elif kind == "neg":
            body.append(f"x{dst}.append(-x{a}[k])")
        elif kind == "scale":
            body.append(f"x{dst}.append(K[{b}] * x{a}[k])")
        elif kind == "mul":
            body.append(f"x{dst}.append(reduce(add, map(mul, x{a}, x{b}[k::-1])))")
        elif kind == "isqrt":
            raise ValueError("isqrt has no series recurrence in this tape")
    for i in range(D):
        body.append(f"x{i}.append(x{tape.rhs[i]}[k] / k1)")
    src = ["def taylor_coeffs(z, n, K, ZERO):"] + head + ["    for k in range(n):", "        k1 = k + 1"]
    src += [f"        {line}" for line in body]
    src.append("    return [" + ", ".join(f"x{i}" for i in range(D)) + "]")
    ns = {"reduce": reduce, "add": add, "mul": mul}
    exec("\n".join(src), ns)
    return ns["taylor_coeffs"]


def compile_values(states, exprs: list, label="values"):
    """f(z, K, sqrt) -> [e(z) for e in exprs], plain value evaluation (RHS, invariants, constraints, forces)."""
    b = _Builder(states)
    nodes = [b.node(sp.sympify(e)) for e in exprs]
    src = [f"def {label}(z, K, sqrt=None):"]
    for i in range(len(states)):
        src.append(f"    x{i} = z[{i}]")
    for kind, dst, a, c in b.ops:
        line = {
            "add": f"x{a} + x{c}", "sub": f"x{a} - x{c}", "mul": f"x{a} * x{c}", "neg": f"-x{a}",
            "scale": f"K[{c}] * x{a}", "const": f"K[{a}]", "isqrt": f"K_ONE / sqrt(x{a})",
        }[kind]
        src.append(f"    x{dst} = {line}")
    src.append("    return [" + ", ".join(f"x{n}" for n in nodes) + "]")
    code = "\n".join(src)
    if "K_ONE" in code:
        one = b._const_id(Fraction(1))
        code = code.replace("K_ONE", f"K[{one}]")
    ns = {}
    exec(code, ns)
    return ns[label], list(b.constants)
