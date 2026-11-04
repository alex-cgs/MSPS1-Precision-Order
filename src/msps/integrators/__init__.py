"""One Stepper interface: step(z, h) -> z_new (spec §9). Taylor additionally exposes coeffs/horner for dense output."""
from .gauss_legendre import GaussLegendre
from .rk import DOP853, RK4
from .symplectic import SplitStepper
from .taylor import TaylorStepper


def make_stepper(cfg: dict, case, arith):
    m = cfg["method"]
    if m == "taylor":
        return TaylorStepper(case, cfg["n"], arith)
    if m == "rk4":
        return RK4(case, arith)
    if m == "dop853":
        return DOP853(case, arith)
    if m == "gl":
        return GaussLegendre(case, arith, cfg["s"], float(cfg.get("solver_tol", 1e-14)))
    if m == "verlet":
        return SplitStepper(case, arith, 2)
    if m == "yoshida":
        return SplitStepper(case, arith, int(cfg["k"]))
    raise ValueError(m)


__all__ = ["TaylorStepper", "RK4", "DOP853", "GaussLegendre", "SplitStepper", "make_stepper"]
