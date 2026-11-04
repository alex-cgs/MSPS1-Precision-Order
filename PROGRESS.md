# MSPS1 progress (agent memory; "continue from PROGRESS.md" is enough to resume)

## Current phase
Code built. E0 done. Dev grids running or analysed, then paper grids. See checklist.

## How to resume
1. Read this file, `CLAUDE.md`, the spec sections touched by the next item.
2. `pgrep -af "msps run|e10_bench"` for running jobs; `tail logs/E*.log`; `.venv/bin/msps status experiments/<file>.yaml`.
3. A run with `msps run` is idempotent: same config = same run_id, finished runs are skipped. Relaunch freely.
4. Analysis: `cd analysis && ../.venv/bin/python eXX_*.py`. Figures: `.venv/bin/python figures/make_all.py`.

## Environment
- `.venv`: main env (Python 3.13.5, pip; no conda on this machine). `.venv-gpu`: JAX 0.12.0.1 CUDA 12 wheels, works on the GTX 1080 Ti.
- 12 threads (Ryzen 5 7600X), 30 GB RAM, 118 GB free disk at start. Other projects on this machine use cron: leave 1 thread free, run at nice 5.
- Tests: `.venv/bin/python -m pytest -m "not slow and not campaign and not gpu"`. GPU: `.venv-gpu/bin/python -m pytest tests/test_gpu.py -m gpu`. Slow pilot regression: `-m slow`.

## Checklist
- [x] Files copied from Drive (`pilot/taylor3b.py`: 6119 bytes here, Drive metadata says 6120; base64 from Drive decodes to 6119, runs identically)
- [x] Pilot rerun (`logs/pilot_rerun/`): all spec §5 numbers reproduced except small differences listed below
- [x] symbolic, arith, integrators, schedules, systems + 19 case cards, diagnostics, models, runner, gpu, io
- [x] Fast test suite passing
- [x] E0 validation: 68/69 checks pass (`results/derived/e0_validation.csv`); the one miss is below
- [x] H100 scripts and cron entries (02:30 `scripts/ops_night.sh`, 04:28 `scripts/ops_cleanup.sh`, tagged `# msps1` in the crontab; remove when E10 is finished)
- [ ] E1 dev done, paper pending
- [ ] E2 dev, paper
- [ ] E4
- [ ] E3 (analysis only, needs E2)
- [ ] E5 dev, paper
- [ ] E9, E8, E6, E11, E10, E7
- [ ] Figures, DIGEST, Drive upload, email, tag results-v1

## Running background jobs
See `pgrep -af "msps run"`. Logs: `logs/E<k>.log`.

## Decisions and deviations from the spec
- **Python `sum` avoided.** Since Python 3.12 `sum()` uses compensated summation on floats. The pilot uses `sum`, so its float64 runs are not plain IEEE rounding on Python ≥ 3.12. The new code uses `reduce(add, ...)`, so f64 is plain IEEE. T_pred at every order is unchanged; c_r moves from the pilot rerun's 4.197 to 4.089 (spec 4.21).
- **Ensemble sizes** follow AGENT_PROMPT (20 dev, 100 paper) rather than spec §8 (100 dev, 1000 paper), as the most recent instruction.
- **E2 cases** follow AGENT_PROMPT (NBN moved to E11). Hénon–Heiles run at both E = 1/8 and E = 1/6.
- **E2 stops a run at E > 1** (saturated). Snapshots and E(t) end there. T_pred unaffected. E(T) is NaN-free only for runs that stay below 1. E5 runs carry on to T on a per-run step.
- **Per-run adaptive mode** (`schedule: adaptive`): same rule as the pilot (h = 2^floor(log2(ρ/8))) with ρ from the run's own last two coefficients, E(t) on the dyadic grid from dense Taylor output. E2 uses shared as primary (same h for all n, as RQ1 requires) and adaptive as a check; E5 uses adaptive.
- **Coarse grid:** dyadic, 256–512 points on [0, T]. Shared-schedule T_pred is still evaluated at every step (pilot definition); only storage is coarse.
- **Perturbation for A(t):** default 1e-40; 1e-20 for nb3_pilot (pilot value); 1e-60 for Lorenz-63, because 1e-20 and 1e-30 (spec §4.7) saturate within T = 80. Linear regime, so size does not matter.
- **Reference protocol:** order 56/400 bits (check 64/500) as the pilot; Lorenz 80/500 (check 90/600), Rössler 64/500 (check 72/600), because amplification is larger.
- **T per case** (spec gives none for most): nb3 120, Lorenz 80, Rössler 800, Hénon–Heiles 800, double pendulum 60, Duffing 300, Lorenz-96 25, FPUT 100, nbN 30.
- **Lifted-variable initial data** that is not rational (figure-eight, Lagrange, Hénon–Heiles px, rings) is stored as a rational to 310 digits, beyond all precisions used. Those cases are excluded from E4 (`skipped_irrational_ic`).
- **DOP853 tableau** is SciPy's float64 coefficients converted exactly, so its order conditions hold to about 1e-17; above 53 bits it is order-limited. Noted in E8. **Yoshida 6, 8:** published 15-digit coefficients, same remark.
- **Gauss–Legendre on the lifted system:** symplectic only in the original variables; it preserves the lifted energy exactly as a quadratic invariant up to solver tolerance. Noted in E8.
- **Not implemented, out of this prompt's list:** Carleman (CARL-N) and the model-truncated ODE surrogate (SUR-n); T_w exists for the sine map only. f80 and Arb exist but are not in the study grids.
- **`ops` first night:** environment check + small benchmark, per AGENT_PROMPT. `scripts/ops_plan.txt` holds `env_small`; change to `full` after a successful night.
- **Disk budget:** `results/raw/` ≤ 20 GB (enforced by the runner); `results/references/` ≤ 20 GB (watched by hand).

## Checks of values marked *verify* in the spec (`results/derived/e0_validation.csv`)
- Lorenz λ: 0.9058 from a 113-bit, n = 24, T = 4000 run, against 0.9056 (Sprott 2003, quoted from memory).
- Rössler λ: 0.0707 from a 113-bit, n = 24, T = 20000 run, against 0.0714 (same source). Agrees within the finite-time spread.
- Figure-eight digits and period: q(6.32591398) closes to 3.0e-8 at 200 bits, n = 32. Consistent with 8-digit initial data.
- Duffing δ = 0.3, ω = 1.2: γ = 0.20, 0.28, 0.29, 0.37, 0.65 give stroboscopic periods 1, 2, 4, 5, 2 with λ < 0; γ = 0.50 is chaotic with λ ≈ 0.12 (dd, n = 24).
- Stone & Leigh 2019 escape criterion (spec §6.1 says *verify*): checked the arXiv text (arXiv:1909.05272v1, abstract, main text and Methods §1 read directly; the rest through a tool summary). The paper gives **no** operational energy-based ejection test: disintegration is idealized as happening inside a "strong interaction region" of radius R = α a_B (or α a_B (1+e_B)), α calibrated from scatterings; integrations end by "standard criteria" cited to Fregeau et al. So the spec's wording does not match the paper. Used here, as this project's own definition: positive energy of body i against the barycentre of the other two, moving outward, and r > 10 × the pair separation (`escape_r_factor` in E5 YAML). Open question 2.
- Hardware facts in spec §11 are not used as numbers; E10 measures.

## Pilot baseline differences (noted, spec not adjusted)
Pilot rerun on this machine: c_r = 4.197 (spec 4.21), 6235 fit points (spec 6207). Everything else equal to the printed digits. New engine on the same setup: c_r = 4.089 (E0's only failed check, tolerance 0.05), caused by plain vs compensated float summation or the fit's sensitivity to rounding-level data; not chased further.

## Wall-time budgets (hours, core-limited) and actual use
| Study | Budget | Actual |
|---|---|---|
| E0 | 0.5 | 0.08 (dev) |
| E1 | 2 | |
| E2 | 12 | |
| E3 | analysis | |
| E4 | 1 | |
| E5 | 20 | |
| E6 | 3 | |
| E7 | 3 | |
| E8 | 6 | |
| E9 | 3 | |
| E10 | 2 h per night | |
| E11 | 10 | |

## Dry-run estimates
- E2 dev (6,328 jobs): sweep 5.8 core-h, adaptive 0.9 core-h, refs/amp/local 0.3 core-h, about 56 MB raw. ~0.7 h wall on 11 workers.

## Open questions for Alexandre
1. The pilot's `sum()` is compensated on Python ≥ 3.12 (see above). Should the paper quote the plain-IEEE numbers from the new engine? The T_pred table is unaffected.
2. Escape criterion: Stone & Leigh give none that is operational. Is the definition above acceptable, and is r > 10 a the factor you want?
