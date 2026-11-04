# Pilot run (reference snapshot)

The first pilot of the method: planar equal-mass three-body problem, Taylor order n × arithmetic precision. Results are summarized in `docs/PROJECT_SPEC.md` §5.

| File | What it does |
|---|---|
| `symbolic_setup.py` | SymPy derivations: Hamilton's equations, polynomialization with u = 1/r, invariants, constraint equation, rational initial data |
| `validate_recurrences.py` | Checks the hand-written Taylor recurrences against SymPy Lie series in exact rationals (orders 0–4) |
| `taylor3b.py` | Taylor integrator of order n with a swappable number type (`float`, `mpfr:<bits>`, `exact`) |
| `experiments.py` | Reference run and shared schedule, reference check, amplification run, local errors, (n, p) sweep, exact-rational run. `--resume` skips finished phases |
| `analysis.py` | Two-constant error-model fit, crossover orders, figures (light and dark) |

Run from this folder:

```bash
pip install numpy scipy matplotlib sympy mpmath gmpy2 python-flint
python3 symbolic_setup.py
python3 validate_recurrences.py
python3 experiments.py      # about 10 minutes on 2 cores; writes results.json and ref.pkl
python3 analysis.py         # writes summary.json and fig/
```

Limitations, which is why the main code should not extend this folder:

- the recurrences are hand-written for three bodies,
- all runs share one step schedule, which fails after a run leaves the orbit,
- there is no provenance or config system.

These files were reassembled after the original session, so they are a new version of the pilot code. A full rerun on 4 November 2026 reproduced every number in spec §5. If a rerun on your machine doesn't, note the differences rather than adjusting the spec.
