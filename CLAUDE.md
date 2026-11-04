# MSPS1 — instructions for Claude Code

This repository is a research project on **truncation order × arithmetic precision in chaotic ODE integration** (Taylor methods, N-body, Lorenz and other nonlinear systems). The full specification is in `docs/PROJECT_SPEC.md`. Read the sections relevant to a task before changing code.

## Ground rules

- The research questions, metric definitions (spec §4.7) and the experiment plan (spec §8) are set by the author. Do not change them silently: propose changes and wait for approval.
- Never invent, round or "smooth" results. Every reported number must come from a stored run whose config and provenance are saved next to it.
- Flag any result that contradicts the pilot baseline (spec §5) or a hypothesis. Do not "fix" it.
- `pilot/` is a frozen reference snapshot. Port ideas from it into `src/msps/`; do not extend it.
- Ask before deleting or overwriting anything under `results/`.
- No parameters hard-coded in code. Configs live in `cases/` and `experiments/` (YAML), and the resolved config is saved with each run.
- Simulation code and plotting code stay separate. Figures read only from `results/`.

## Workflow

- Run the fast test suite (`pytest -m "not slow and not campaign"`) before reporting a task as done.
- Before launching any long campaign: do a 1–2 configuration dry run, report the estimated total time and disk use, then launch it resumable (checkpoints, skip finished runs) with `nohup`, and say how to monitor it.
- The H100 (`ssh ops`) may only be used between 02:30 and 04:30. Jobs there must stop or checkpoint by 04:25 and sync results back afterwards (spec §11).
- Every GPU kernel needs a CPU reference implementation of the same algorithm, and a test comparing the two.

## Layout (see spec §9)

`src/msps/{arith,symbolic,systems,integrators,schedules,gpu,diagnostics,models,runner,io}`, `cases/`, `experiments/`, `results/{references,raw,derived}`, `analysis/`, `figures/`, `bench/`, `pde/`, `tests/`, `pilot/`, `paper/`.

## Environments

- `environment.yml`: main CPU env (numpy, scipy, sympy, mpmath, gmpy2, python-flint, matplotlib, pandas, pyarrow, pytest, hypothesis).
- `environment-gpu.yml`: adds JAX (CUDA) or CuPy. Use a separate env per machine.
- `environment-fenics.yml`: DOLFINx from conda-forge, for `pde/` only.
