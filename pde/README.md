# PDE extension (planned, spec §6.3, study E12)

Out of scope for the current experiment run. Planned cases:

- **Viscous Burgers** on a periodic domain, with the Cole–Hopf exact solution as an oracle (no chaos).
- **Kuramoto–Sivashinsky** at L = 22 (value to be verified), ETDRK4 baseline (Kassam & Trefethen). Truncation knob: number of modes N; statistic: energy spectrum.
- **Cahn–Hilliard** from the DOLFINx demo: mixed P1 elements, theta = 0.5, Newton solver. Knobs: mesh size, degree, time step, float32 vs float64.

DOLFINx has no arbitrary precision, so the PDE path covers the float32/float64 axis and the discretization knobs only. Use `environment-fenics.yml`.
