"""Step-size rules (spec §7.1). All return exact Fractions, so step times stay exact."""
from __future__ import annotations

import math
from fractions import Fraction


class FixedStep:
    def __init__(self, h: Fraction):
        self.h = Fraction(h)

    def next_h(self, t, z, coeffs, stepper) -> Fraction:
        return self.h


class AdaptiveDyadic:
    """h = 2^floor(log2(rho/safety)), capped; rho from the last two coefficients (Jorba–Zou style).

    The pilot's rule. Used by reference runs to build shared schedules, and by each run for itself in per-run mode.
    """

    def __init__(self, safety: float = 8, cap: Fraction = Fraction(1, 2), floor_exp: int = -400):
        self.safety = float(safety)
        self.cap = Fraction(cap)
        self.floor_exp = floor_exp

    def next_h(self, t, z, coeffs, stepper) -> Fraction:
        rho = stepper.radius(coeffs)
        if not math.isfinite(rho):
            return self.cap
        e = math.floor(math.log2(rho / self.safety)) if rho > 0 else self.floor_exp
        e = max(e, self.floor_exp)
        return min(Fraction(2) ** e, self.cap)


class SharedSchedule:
    """Steps built once by the reference run and reused by every run (fails once a run leaves the orbit, spec §5)."""

    def __init__(self, steps: list):
        self.steps = steps
        self.k = 0

    def next_h(self, t, z, coeffs, stepper) -> Fraction:
        h = self.steps[self.k]
        self.k += 1
        return h

    def exhausted(self) -> bool:
        return self.k >= len(self.steps)


def make_schedule(cfg: dict):
    kind = cfg["kind"]
    if kind == "fixed":
        return FixedStep(Fraction(str(cfg["h"])))
    if kind == "adaptive":
        return AdaptiveDyadic(float(cfg.get("safety", 8)), Fraction(str(cfg.get("cap", "1/2"))))
    raise ValueError(f"unknown schedule {kind}")
