# Notes: statistical prediction and confidence regions (planned paper 2, spec §14)

Not implemented. Plan:

- Treat numerical error as an equivalent initial uncertainty ε(n, p), justified by shadowing.
- Propagate ensembles and report regions containing 95% of the ensemble after time t. For N-body use relative quantities: pair distances, which body escapes.
- Three regimes: (1) linear, covariance through the tangent map, ellipsoids; (2) nonlinear, Monte Carlo quantile or density regions; (3) saturated, regions converge to the invariant measure or to escape-outcome probabilities.
- Validate by coverage: the reference falls inside the 95% region about 95% of the time.
- Width model: log W ≈ log ε(n, p) + λ t, until saturation.

## Data already stored for this

Every `traj` and `stats` run writes `states.parquet`: the full lifted state of each member at 20 equally spaced output times (float64 values), next to the exact rational initial condition in `meta.json`. Sweep runs stop once E > 1, so their snapshots end there; `stats` runs (E5) go to their full horizon on a per-run adaptive step and keep all 20.
