# MSPS1: Taylor order × arithmetic precision in chaotic integration

How long does a numerical solution of a chaotic system stay correct, as a function of the Taylor order *n* and the arithmetic precision *p*? This repository holds one integrator code path. It runs unchanged in IEEE f16, f32 and f64, in double-double, in MPFR at any bit count and in exact rationals, on polynomialized ODEs: the N-body problem, Lorenz-63, Rössler, Hénon–Heiles, Lorenz-96, FPUT, Duffing, the double pendulum and iterated maps. The repository also holds the experiment configs, the analysis, the derived tables and the figures.

- **Specification:** research questions, metrics and the experiment plan (E0–E13) are in [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).
- **Results:** the summary is [`results/DIGEST.md`](results/DIGEST.md). Tables are in `results/derived/`, figures in `figures/out/`.
- **Run log:** deviations from the spec, open questions and the compute budget are in [`PROGRESS.md`](PROGRESS.md).
- **Status:** studies E0–E11 are done (tag `results-v1`). E12 (PDE) and E13 (confidence regions) are not started. The H100 part of E10 was not run.

## Layout

| Path | Contents |
|---|---|
| `src/msps/` | The `msps` package: `symbolic` (polynomialization, Taylor tapes), `arith` (number back-ends), `integrators` (Taylor, Gauss–Legendre, Yoshida, RK4, DOP853, maps), `schedules`, `systems` (case loader, initial conditions), `gpu` (batched JAX kernels with a numpy reference), `runner` (job queue, references, provenance, CLI), `io` |
| `cases/` | One YAML case card per system (parameters, initial-condition ensembles, reference protocol) |
| `experiments/` | One YAML file per study, with `dev` and `paper` grids. `superseded.yaml` lists cases that analysis skips; `reference_variants.yaml` maps high-precision reference variants back to their parent case |
| `analysis/` | One script per study. Reads `results/raw` and `results/references`, writes `results/derived/` (CSV plus a Markdown copy) |
| `figures/` | Figure scripts. They read `results/derived/` only and write `figures/out/` (PNG and PDF, light and dark) |
| `bench/` | E10 hardware benchmark (CPU numpy against the GPU) |
| `scripts/` | `reproduce.sh` (full pipeline), E10 bitwise diagnostic, replay diagnostics, H100 night scripts |
| `tests/` | Fast unit and property tests, GPU-against-CPU tests, slow pilot regression |
| `pilot/` | Frozen snapshot of the pilot code (spec §5). Not extended; see `pilot/README.md` |
| `pde/`, `paper/` | Placeholders for E12 and the manuscript |

## Setup

Python ≥ 3.11 (the runs used 3.13.5). The main environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e . pytest hypothesis
```

With conda, `conda env create -f environment.yml`, then `pip install -e .` inside it, does the same. `gmpy2` and `python-flint` need wheels for your platform, or the GMP and MPFR headers to build them.

The GPU environment, needed only for E10 and the GPU tests, uses JAX with CUDA 12 wheels:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -e . "jax[cuda12]" pytest hypothesis
```

## Tests

```bash
.venv/bin/python -m pytest -m "not slow and not campaign and not gpu"   # fast suite, about a minute
.venv-gpu/bin/python -m pytest tests/test_gpu.py -m gpu                 # every GPU kernel against its CPU reference
.venv/bin/python -m pytest -m slow                                      # pilot regression (spec §5)
```

## Reproducing the results

`results/raw/` (about 2 GB) and `results/references/` (about 8 GB) are not in git. Everything in `results/derived/` and `figures/out/` can be rebuilt from the configs:

```bash
nohup scripts/reproduce.sh all > logs/reproduce.out 2>&1 &
```

The script runs four stages in order. A single stage can be run on its own with `scripts/reproduce.sh <stage>`:

1. **`runs`:** every study config, paper grids, `--workers $WORKERS` (default 7) at nice 5, then a retry pass for jobs that hit the memory cap.
2. **`e10`:** the hardware benchmark on the GPU and on one CPU core, plus the bitwise CPU–GPU comparison.
3. **`analysis`:** all analysis scripts, which write `results/derived/`.
4. **`figures`:** `figures/make_all.py`, which writes `figures/out/`.

**Cost.** The original campaign used about 430 core-hours on a Ryzen 5 7600X, plus about 4 GPU-hours on a GTX 1080 Ti and 3 hours on one CPU core for E10. The slowest single jobs are the N = 8 and N = 10 references in `e11_high_n.yaml`, at up to about 10 h each, reference and check together. They run without checkpoints, so killing one loses it. One job never run in the original campaign, `amp_fix2` in `e11_high_n_t8.yaml`, would take about 15 h. No reported number depends on it.

**Resuming.** The pipeline is idempotent. A run's ID is a hash of its resolved config, and finished runs are skipped, so the script can be stopped and relaunched at any time. To follow progress:

```bash
tail -f logs/<study>.log
.venv/bin/msps status experiments/<study>.yaml --mode paper
```

**Single studies.** To run one study:

```bash
.venv/bin/msps dryrun experiments/e02_sweep.yaml --mode paper              # time and disk estimate from sample jobs
.venv/bin/msps run experiments/e02_sweep.yaml --mode paper --workers 7 --log logs/e02_sweep.log
.venv/bin/python analysis/e02_sweep.py
```

E3 has no runs of its own: `analysis/e03_model.py` fits and tests the error model on E2 data. `--mode dev` runs the smaller development grids (20 initial conditions instead of 100).

**Provenance.** Each run directory `results/raw/<study>/<run_id>/` holds `meta.json` next to its Parquet data. `meta.json` records the resolved config, the git commit and whether the tree was dirty, package and MPFR versions, the host, CPU (including FMA support), GPU and wall time. References are stored per case and IC in `results/references/` (`ref.pkl` plus `meta.json`), and each one is checked against a higher-order, higher-precision rerun to 1e-30.

**Bitwise reproducibility.** f64 and dd results are bit-identical between the numpy CPU path and the JAX GPU path. f32 results are not, because GPU f32 division is not correctly rounded (`results/derived/e10_gpu_elementary_ops.csv`). The code never calls Python's `sum()` on floats, because it is compensated on Python ≥ 3.12 and so would not be plain IEEE arithmetic.

## Diagnostics behind specific claims in the digest

Run these from the repository root after the runs exist. Each prints its result; the stored copies are in `results/derived/`.

| Script | Output | Claim |
|---|---|---|
| `scripts/nb3_underflow_replay.py` | `diag_nb3_underflow.txt` | 3-body step underflows at low n come from u = 1/r drifting from the positions |
| `scripts/f32_overflow_replay.py` | `diag_f32_overflow.txt` | f32 Taylor coefficients overflow on the 3-body ensemble |
| `scripts/dd_overflow_replay.py [ORDER]` | `diag_dd_overflow.txt` | dd coefficients leave its range at very small steps (N = 8) |
| `scripts/ref_diff_profile.py CASE IC` | `e11_ref_diff_profile.csv` | When an N ≥ 8 reference fails its check |
| `scripts/e10_bitwise_diff.py [BATCH]` | `e10_bitwise_diff.csv`, `e10_gpu_elementary_ops.csv` | CPU–GPU bitwise comparison (GPU environment) |
| `analysis/e04_hh_degree.py` | `e4_hh_map_degree.csv` | The degree of the Hénon–Heiles Taylor map is n/2 + 1 |

## H100 runs (optional)

`scripts/ops_night.sh [env_small|full]` and `scripts/ops_cleanup.sh` run E10 on a remote H100 reached as `ssh ops`. Per spec §11, the machine may only be used from 02:30 to 04:30: the first script is started by cron at 02:30 and the second at 04:28. Both check for a visible NVIDIA GPU before copying anything. The `ops` host had none when tried, so no H100 data exists.
