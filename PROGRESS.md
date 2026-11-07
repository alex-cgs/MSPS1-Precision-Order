# MSPS1 progress (agent memory; "continue from PROGRESS.md" is enough to resume)

## Current phase
All studies E0-E11 finished on 2026-11-07 at 18:04; nothing is running. Tests pass (fast, GPU, slow). `results/DIGEST.md` written, figures and tables final, email sent, tagged `results-v1`. Open: the author's decisions below, and copying the figures and CSVs into Drive (checklist).

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
- [x] H100 scripts written. **The author dismissed the `ops`/H100 tests on 2026-11-05: E10 runs locally.** Cron lines removed; `~/msps1` (306 MB) left on ops, nothing running
- [x] Bibliography saved (`MSPS1 bibliography.csv`, 43 entries)
- [x] Code snapshot in Drive: `MSPS1/code` (folder id 1w6p--Df2pI5MirUF7CcJ4XovKlRZJMvM), commit ee91ef0, asked for by the author on 2026-11-05. Later code changes are not in it; `results/` is not in it
- [x] E1 paper finished 23:00 on 2026-11-05: 78 done, 6 failed, all `nbN_circle:N_10` (3 references, 3 local); see flags. `analysis/e01_local.py` run
- [x] E2 paper done: 885 failed jobs from 8 reference checks are covered by the fix groups (`rossler:r2`, `nb3_ensemble:r5`), so all 100 ICs are in the tables. The 2,518 pending rows are the superseded base `nb3_ensemble` grid
- [x] E3 final, on all 100 ICs (50 train, 50 test)
- [x] E5 paper done: 14,860 done, 175 failed, the 8 known ICs before their fix groups. `analysis/e05_stats.py` run
- [x] E9 paper: 3,054 done, 300 failed by design (`sine_map` qr-p)
- [x] E8 364, E6 5,940 (2 memory failures retried), E7 1,600, E4 44: paper done, tables in `results/derived`
- [x] E11 N <= 6 paper done (42 failed on `N_6` #3, covered by `N_6_hp`). N = 8, 10 done at T = 8, 5 ICs per family, 3 of them through hp references, plus the T = 30 bridge. E11's 174 pending and E11hi's failed rows are grids that depend on failed references, left unrun by design. `amp_fix2` (amplification for `nbN_cluster:N_8_T8_hp` #4) deliberately not run: about 15 h on one core against the 6 h job limit, and no reported N >= 8 number uses amplification
- [x] E10 local (CPU one numpy core, GTX 1080 Ti); H100 dismissed
- [x] Interim findings emailed to the author on 2026-11-05 at their request, subject "MSPS1: interim findings (not the final digest)"
- [x] Figures (32 PNG, 32 PDF, light and dark), `results/DIGEST.md`
- [x] Digest email sent 2026-11-07, subject "MSPS1: experiment digest"; repository tagged `results-v1`
- [ ] **Drive upload incomplete, needs the author.** Drive `MSPS1/results` (folder id 1ykFPbhMYGr1DrQFv1pw3kTZXPVSIUxLE) holds `DIGEST.md` and `PROGRESS.md` only. The connector takes file content inline, so the 32 PNG (8.3 MB) and 35 CSV (20 MB) cannot go through it, and this machine has no Drive mount or CLI. Retyping the small CSVs by hand was rejected as a transcription risk on research data. All 67 files are ready in `results/drive_upload/` (git-ignored) to drag into that Drive folder. The digest with its figures is also on a private page: https://claude.ai/artifact/9FpUhsNF3CPN478UEwFiYp
- [ ] Drive `MSPS1/code` is still the ee91ef0 snapshot. Since then: small changes to `runner/queue.py`, `integrators/gauss_legendre.py`, `io/__init__.py`, plus analysis, figures, tests, experiments, cases; `bench/`, `scripts/` not included. The tag `results-v1` holds the current code; the repository has no remote

## Running background jobs
None. History, in order (logs: `logs/E<k>.log`). Started 2026-11-06 at 12:26, 11 threads:
- `scripts/paper_chain2.sh` (replaces `paper_chain.sh`, stopped by PID): E5, E9, E8, E6, E7, E4; 7 workers.
- E2 fix groups (`refs_fix` etc, `rossler:r2`, `nb3_ensemble:r5`; log `logs/E2fix.log`), then E11 dev, now N <= 6 and Lorenz-96 only; 2 workers.
- `experiments/e11_high_n.yaml` (study E11hi): route 1, order 80/600 at T = 30, N_8_hp then N_10_hp, IC 0; 1 worker. No checkpoints: a kill loses the reference.
- 16:25: `experiments/e11_high_n_t8.yaml` (route 2, T = 8, 3 families, N = 8, 10, ICs 0-2), 2 workers; E11 paper for N <= 6 once the profiler ends, 1 worker. E2 fix done: all 8 ICs pass (worst 1.6e-39), 885 of 885; E11 dev done, 434 of 434.
- 2026-11-07 00:45: `e11_high_n_t8.yaml --mode dev --groups amp local sweep` (N = 8, 10, T = 8, ICs 0-2), 6 workers, log `logs/E11t8sweep.log`; `--mode paper --groups t8_n8 t8_n10` (references ICs 3, 4), 2 workers. After those: `--mode paper` for the rest and the fix groups. E4-E9 and E11 N <= 6 paper done.

- 2026-11-07 04:30: `e11_high_n.yaml --mode paper --groups t8_cluster_fix` (`nbN_cluster:N_8_T8_hp` #4 reference, expect about 8 h, log `logs/E11t8fix.log`); the ICs 3, 4 sweep plus the `N_10_T8_hp` #2 fix, 6 workers, from a partial copy of `e11_high_n_t8.yaml` that leaves out cases whose reference is not ready (same run IDs, so the full YAML skips them later). Also `t8_cluster_fix10` (`nbN_cluster:N_10_T8_hp` #4, log `logs/E11t8fix10.log`). `N_10_T8_hp` #4 passed at 1.76e-45; `amp_fix3`, `sweep_fix3` done (41 of 41). Still to run with the full YAML once `N_8_T8_hp` #4 is done: `amp_fix2`, `sweep_fix2`.
- 2026-11-07 09:01: E10 `local_gtx1080ti --size full` with `.venv-gpu`, log `logs/E10_gpu.log`. One nice-5 reference job on a 12-thread CPU was running; the GPU was otherwise free. Done 12:43, 13,348 s, all 108 + 18 rows.
- 12:45: E10 CPU, `--backend numpy --label local_cpu_numpy --size cpu --accuracy-orders 16 24`, log `logs/E10_cpu.log`. Batch stops at 1e4 (1e5 is about 15 h on one numpy core); accuracy skips n = 8 (nb3 n = 8 took 2,880 and 6,117 s on the GPU).

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
Core-hours are the sum of job wall in `results/queue/<study>.sqlite`, dev and paper, finished plus failed, on 2026-11-07. A retry overwrites the earlier wall, and runs killed by hand are not stored, so these are lower bounds. Total 427 core-h (352 finished, 75 failed), plus E10.

| Study | Budget | Actual |
|---|---|---|
| E0 | 0.5 | 0.19 core-h |
| E1 | 2 | 28.5 core-h (24.8 on failed N = 10 references); paper 12.4 h elapsed: over budget |
| E2 | 12 | 64.1 core-h; paper 12:30 to 19:46 (7.3 h elapsed) |
| E3 | analysis | |
| E4 | 1 | 0.70 core-h |
| E5 | 20 | 175.6 core-h; paper 25.9 h elapsed on 7 workers, 2026-11-05 19:46 to 2026-11-06 21:37: over budget, 16.7 h of it before the `henon_n2` cap |
| E6 | 3 | 4.64 core-h |
| E7 | 3 | 0.46 core-h |
| E8 | 6 | 1.30 core-h |
| E9 | 3 | 0.05 core-h |
| E10 | 2 h per night (H100) | local: GTX 1080 Ti 13,348 s, CPU about 3 h on one core |
| E11 + E11hi | 10 | 101.4 + 50.1 failed = 151.5 core-h: over budget, because N >= 8 references needed new protocols |

## Second incident, E4 dev, 2026-11-05
E4 sat idle from 04:21 to 09:25: `GNU MP: Cannot allocate memory` aborted a worker, and `mp.Pool.imap` waits forever for a job lost with its worker. Fixed: `run_pool` in `runner/queue.py` fails that job, replaces the worker and enforces an optional `job_wall_limit_s` (study YAML). `run_exact` now stops before a step predicted to pass `max_wall` or `max_bits` (bits and walls of completed steps unaffected). Nine E4 runs finished before the change are valid and kept; some of them ran past `max_wall` (up to 689 s against 200 s) and have more steps than the new rule would allow.

## Other defects found on 2026-11-05
- **`IteratedMap.step` skipped `post_step`,** so `qr-p` map iterates were exact rationals and doubled in size per iteration. E9 sat for 56 minutes on seven workers before I saw it. Fixed in ee91ef0 with a test. No `qr` map run had been stored, and the other back-ends have a no-op `post_step`, so no stored result changes.
- **`t_end` overwritten.** `execute_job` stored a wall-clock stamp under the name of the run's simulated `t_end`. Fixed in 4860044; the stamps are now `clock_start`, `clock_end`. Older runs hold no simulated end time, and `analysis/common.sim_t_end` reports NaN for them. T_pred, E(t) and all other columns are unaffected.
- **`nb3_ensemble` card had no `tolerances`.** 1,300 E2 jobs failed with a KeyError; rerun after adding the same tolerances as every other card.
- **E2 memory cap.** 81 jobs on the longest references hit MemoryError at 2.2 GB per worker. All completed on retry with 4 workers at 5 GB. Paper grids need the same second pass.
- **Duplicate fixed ICs in E1.** Cases with a fixed IC (for instance `nb3_pythagorean`) run ic 0, 1, 2 with identical results, so their `ics` count in `e1_local_law` overstates independent samples. Medians are unaffected.

## Flags against hypotheses and the plan (not adjusted)
- **Mechanism, nb3 step_underflow at low n (found 2026-11-07, replay in f64, 30 ICs, t <= 300, no run stored):** the lifted variable u = 1/r desynchronises from the positions. All 73 underflows at E5's cap 1/2 have |u r - 1| > 1 (u r about 2e8 to 6e8 on the ones printed, bodies apart: rmin 2.5 to 3.7 on 3 of 4 printed), only 1 has rmin < 1e-6, at n = 16. Counts by n: 4: 29, 6: 20, 8: 16, 12: 7, 16: 1 of 30, close to the E5 fractions (`logs/nb3_underflow_replay.txt`). Same in mpfr-113 on ICs 0, 1, 2, so truncation, not rounding. So the low escape fraction at low n in E5 is largely runs dying from a polynomialization artefact before they can escape, not a wrong escape law. They count as no escape in `esc_frac`. Not fixed: re-synchronising u each step would be a method change, the author's decision.
- **Mechanism, exponent range:** Taylor coefficients are not scaled by h, so |c_k| about rho^-k leaves the format's range. f32 (1e38): on nb3 at n = 32, 24 of 30 ICs go non-finite before t = 6, median t about 1.1, and 99% of E2 runs blow up. This explains the earlier unexplained fall of f32 T_pred with n (tol 1e-3: 4.17 at n = 8 to 1.89 at n = 32) and E5's f32 escape fraction falling to 0. dd (double range): N = 8 #0 at n = 32 goes NaN at step 1127, h = 2^-35, largest finite coefficient 5e294, so T_pred 4.76 against 24.13 for mpfr-113. dd under 0.9 of mpfr-113 on 9% of nb3 ICs at n = 32, 0 below n = 24; dd is not a clean 106-bit stand-in at high n through close encounters. MPFR is unaffected. Not fixed: scaling coefficients by h^k would change the tape, the author's decision.
- **E11 bridge, T = 30:** `N_8_hp` #0 f64 plateau 3.59, 113 bits 24.13 at n = 32; `N_10_hp` #0 f64 5.84, 113 bits 14.85, 0% censored, one IC each. `N_10_hp` passes its check at 4.94e-31, 41,038 steps, 12,339 + 15,601 s. `N_10_T8_hp` #2 passes at 1.62e-49. E1 N = 10 at T = 8: slope -1.127, rho/h 13.4, 3 ICs.
- **Leftovers kept on disk, nothing deleted (ask before removing):** `E*_dry` runs, 44 older E0 runs, E10 tiny and `dry_numpy` runs, superseded E1 runs and references, failed N >= 8 references and `.tmp` files. On ops: `~/msps1` (306 MB).
- **E5 vs E2 T_pred:** E5 is higher (f64, n = 16 medians: Rossler 269 against 234.5, 93% of ICs higher; Lorenz 20.0 against 19.4; nb3 44.1 against 40.2). E5 tests E(t) only on the reference's coarse output grid and uses its own adaptive schedule, cap 1/4; E2 tests every shared step. Subsampling E2's stored E(t) to Rossler integer times gives median 256, so the grid explains about 22 of the 34.5. Do not compare the two; E5's T_pred is for contrast with statistics only.
- **E5, RQ3 (paper):** H3 holds on Lorenz and Rossler: lambda within 0.0031 and 0.0012 of reference and W1 close to the split-half floor (at n = 2 up to about 2-3 times it on Rossler) at every (n, p) including f32, where Lorenz T_pred is 0.25 (one grid step). H3 fails on nb3_ensemble at low n: escape fraction 0.12 at n = 4 and 0.35 at n = 8 against reference 0.84, same in all of f64, dd, 113 bits, so truncation, not precision; f32 worse with n (0.34 at n = 6, 0.00 at n = 24, 32), mostly step_underflow. Hénon-Heiles n = 2 wrong in all arithmetics (lambda error > 0.1, orbit runs away), mixed 60 s and 3,600 s caps.
- **E6:** Hénon-Heiles E = 1/8 section overlap stays near 0.7 up to n = 16 in f64 and dd with energy error down to 1e-15, against the reference dd n = 24. Likely the finite chaotic segment, not spurious dynamics, since overlap is flat in n and p; no noise floor computed for it, so do not read 0.7 as damage. Duffing gamma = 0.28, 0.29: f32 gives 3 and 9-10 distinct values where f64 gives 2 and 4, at n = 6 to 16, regime changed by precision alone.
- **E8, single IC:** at f64 high-order families reach similar plateaus: Lorenz 19.4-20.1, pilot 70.4-78.8, a 12% spread with Yoshida highest. RK4 stays lower (14.3, 40.4). Hénon-Heiles: Taylor, DOP853, GL censored at 800, RK4 and Yoshida 546. Taylor is cheapest on Lorenz and Hénon-Heiles under 0.3 s; Yoshida on the pilot. Walls measured under load.
- **E10, RQ8 (2026-11-07), CPU one numpy core against GTX 1080 Ti; H100 not tested:** f64 and dd are bit-identical on every finite value, 100 members, n = 8, 16, 24, through 3,200 steps at h = 1/64 (`results/derived/e10_bitwise_diff.csv`). The nb3 hash mismatches in `e10_reproducibility.csv` (batch >= 10) are only the sign of NaN on members that blew up under the fixed step, 0x7ff8... against 0xfff8...; 5 to 10 of 100 are non-finite after 32 steps. The fingerprint hashes raw bytes, so read that table with this in mind. f32 is not reproducible: 4 or 5 of 100 Lorenz members differ after 32 steps, 71 to 75 after 3,200, by up to 35. Likely cause: GPU f32 division is not correctly rounded, 29.8% of random divisions differ, up to 2 ulp, 24.2% for division by an integer; f32 add and mul and all f64 operations match on 2e6 operands (`e10_gpu_elementary_ops.csv`). The kernel divides by k, so this is the likely source; replacing the division was not tried. Flush-to-zero was tested and ruled out.
- **E11, ICs 3, 4 at T = 8 (2026-11-07):** all pass at 56/400 except `nbN_cluster:N_8_T8` #4 at 1.08e-30 against 1e-30 (135,246 steps, 7,656 + 9,701 s). tol_check not relaxed: retried through `nbN_cluster:N_8_T8_hp` (80/600, check 90/700). `nbN_cluster:N_10_T8` #4 fails at 1.25e-23 (16,687 steps, 1,479 + 1,867 s), retried through `nbN_cluster:N_10_T8_hp`. Both pass: `N_8_T8_hp` #4 at 7.10e-53 (132,568 steps, 16,043 + 18,853 s), `N_10_T8_hp` #4 at 1.76e-45 (16,458 steps, 3,683 + 4,052 s).
- **E11, N = 8, 10 routes (2026-11-06, ICs 0-2):** T = 8 at 56/400 passes 17 of 18 references (worst passing 4.4e-33); `nbN_circle:N_10_T8` #2 fails at 5.9e-27, retried through `N_10_T8_hp`. Higher precision at T = 30 (80/600, check 90/700): `nbN_circle:N_8_hp` #0 passes at 1.84e-44, 15,170 steps, 2,539 + 3,267 s; `N_10_hp` #0 still running. N = 8, 10 enter E11 at T = 8, so their T_pred is censored at 8, against T = 30 for N <= 6; `N_8_hp` #0 runs the sweep at T = 30 as a one-IC bridge. E11 paper: `nbN_circle:N_6` #3 fails at 2.05e-29, variant `N_6_hp` to follow.
- **E11, N = 8 too (2026-11-06, base protocol 56/400, T = 30):** `rings_N_8` #0, 1, 2 fail (3e-18, inf, 7.9e-26), `nbN_circle:N_8` #0 fails (1.15e-22), #2 passes (5e-43), #1 hit the 6 h limit, `rings_N_10` all three hit the 6 h limit. Close encounters reach dmin 1e-7 to 1e-6. N <= 6 all pass. The author said to try both more precision and a shorter T for N = 10, falling back to reporting up to N = 8 with an explanation; N = 8 needs the same remedy before it can be reported.
- **E5, Henon-Heiles n = 2:** energy drifts over the escape energy (H 0.1667 to 0.21 by t about 106 on IC 0), the orbit runs away and rho -> 0, so t stalls near a limit: 60 s reaches 106.708, 3600 s reaches 106.727. All 77 stored E_1_6_r2 n = 2 runs ended in `wall_budget` at 3600 s with median t_end 89 of 5000. New group `henon_n2` caps these at 60 s (`max_wall` is outside run_id). Stored 3600 s runs stay. This saves about 700 core-h. Five f32 runs crashed writing a checkpoint with a NaN twin; under 300 s they never reach a checkpoint.
- **Correction, E2 crossover law:** the earlier "bits per order 3.54-3.76 against predicted 3.54-3.75" compared `n_star_model` with a prediction from the same local fit, so it was close to circular. The table now also has `empirical_bits_per_order` from the empirical n*, on cells where at least half the ICs reach a plateau. On paper data: Lorenz 3.5 (predicted 3.59), Rossler 3.1 (3.59), Henon-Heiles E = 1/6 4.0 (3.71), 3 points each on an even-order grid. Empirical and model n* agree within about 2 orders at 24, 53, 80 bits.
- **E2 paper, failed references:** rossler #26, 35, 47, 58, 63, 91, 99 (max diff 1.5e-30 to 4.4e-29 against tol 1e-30) and nb3_ensemble:r4 #65 (3.45e-21). 885 dependent jobs failed with them. The check was not relaxed. Plan told to the author: recompute these ICs with higher-precision references in a new variant, same check, once cores are free. Done: all 8 pass through `rossler:r2` and `nb3_ensemble:r5` (worst 1.6e-39), 885 of 885 jobs.
- **E1 paper, N = 10:** `nbN_circle:N_10` has no data. Reference #0 fails its check (max diff 4.99e-10 against 1e-30); others hit MemoryError or a worker abort after hours. Asked the author whether to try more precision, a shorter T, or stop at N = 8. E11 includes N = 8, 10 references and may fail the same way.
- **E2 paper, explained later by the exponent-range mechanism above:** `nb3_ensemble:r4` in f32 at tol 1e-3: median T_pred falls from 4.39 at n = 8 to 1.91 at n = 32, and is lower on 95 of 99 ICs pairwise. No such trend on Lorenz, Rossler or Henon-Heiles E = 1/8.
- **E1 paper, schedule:** rho/h of about 12-14 follows largely from the schedule, which puts h between rho/16 and rho/8; do not read it as a property of the systems. Kepler e = 0 and Lagrange are outliers (slopes -1.61, -1.73, rms 1.35); cause not checked.
- **E2 queue:** the 2,518 pending rows are the superseded base `nb3_ensemble` grid, not part of the paper grid.
- **E4, RQ9:** Hénon–Heiles grows at 0.56–0.73 of (d−1)n+1. The exact degree of its order-n Taylor map is n/2+1 (`analysis/e04_hh_degree.py`), and measured growth follows that: 7.24 against 7 at n = 12, where the law gives 13. So the law is an upper bound, reached only when no such structure lowers the degree. N-body 0.86–1.00, Lorenz 0.81–0.93.
- **E3, H2 (dev grid, superseded by paper numbers in `e3_model_out_of_sample.csv` and DIGEST: nb3 3.24 decades, 25% within one):** on `nb3_ensemble:r4` the model does not transfer across ICs: out-of-sample rms 2.84 decades, 32% of points within one decade; the FTLE variant is no better (2.70). Per-IC fits reach about 1.2 decades on most ICs, with c_t from 8.1 to 19.7 and c_r from 3.9 to 11.7. The other systems test at 0.75–1.35 decades, and the FTLE variant improves Lorenz (0.53) and Rössler (0.76). Dev grid, 20 ICs.
- **E9, RQ6:** `sine_map` has no `qr-p` runs: 30 jobs fail with `qr-p has no sin`, since sin of a rational is not rational. They stay failed unless the author wants a definition. `tent_map:mu_2` at 53 bits and above is fully censored, because the 30-bit dyadic ICs are exact there and the orbit matches the reference until collapse.
- **E5 (dev grid, superseded by the mechanism flag above):** at n = 8, 11 of 20 `nb3_ensemble:r4` runs end in `step_underflow` under both f64 and 113 bits, so it is not a float-range effect. It reduces their escape fraction (0.40 against 0.85 at n = 32). Likely the per-run step collapsing into a close encounter at low order; cause not confirmed.

- **E6, RQ4:** distinct stroboscopic values overstate regime changes. At n = 3, 4 with gamma = 0.20, lambda stays near -0.15 and x moves by under 0.06, yet all 300 values differ at 1e-6: the fixed point drifts slowly, which fits non-conservation of the lifted forcing (cause not confirmed). `regime_changed` (58% of rows, dev) counts these; `spurious_chaos` and `spurious_order` use the sign of lambda only. Spurious chaos appears at n = 2 for every non-chaotic gamma except 0.65, and at n = 3 for gamma = 0.37 and 0.65. Lambda values are for f64, dev grid.
- **E9, RQ6:** rounding to 2^-53 beats f64 on Lorenz (T_pred about 23 against 19 at top n) and loses on the pilot (about 64 against 70). Expected direction, since absolute rounding is finer than relative at |x| near 30 and coarser for large u_ij at close encounters, but the mechanism is not tested separately.
- **E7, RQ5:** paper: Spearman 0.914, AUC 0.954 at n = 16, 1,266 of 1,600 points. Dev grid gave 0.90, 0.98 on 77 points.
- **E8 Gauss–Legendre:** s = 4, 6 failed at construction (repeated root in `tableau`), fixed in 68eeb15 and rerun. Stored s = 2 runs stay valid: the tableau is unchanged at working precision.

## Dry-run estimates
- E11hi, N = 8, 10 at T = 8, ICs 0-2 (dry run 2026-11-06): amp 8.19, sweep 13.08, local 0.01 core-h, 45 MB; about 35 core-h for 5 ICs. Observed: 13,302 s on 6 workers for ICs 0-2.
- E5 dev 2.36 core-h, E9 0.06, E8 5.35, E6 2.51, E7 0.02 (`logs/dry_*.log`, 4 samples per group). E11 did not finish in 25 min: N = 6 references take 5 to 30 min and N = 10 are over 1.5 h each, so its paper grid may pass its 10 h budget.
- E2 dev (6,328 jobs), first estimate: sweep 5.8 core-h, adaptive 0.9 core-h, refs/amp/local 0.3 core-h, about 56 MB raw. ~0.7 h wall on 11 workers. References were underestimated (see incident).
- E2 dev sweep, redone on r4 references with 12 samples per group (`logs/E2_dry2.log`): sweep 0.42 core-h, adaptive 0.02 core-h, 32 MB. Sample walls stay under 2.1 s, so the long-lived high-n, high-p tail is probably missed; treat as a lower bound.
- E4 dev (dry run hit my 150 s limit, rerun): 4.35 core-h total, 0.1 MB.

## Incident, night of 2026-11-05
- Nine `nb3_ensemble` references ran from 00:07 to 02:35 without finishing, each near 2.4 GB, and took the machine to 247 MB available with swap full. Stopped by hand, with E4 removed from the chain. Sampled before stopping: 4–87% of T after 2.4 h, close encounters near 3e-4.
- **Cause (confirmed by replaying #9 at 56/400 and 64/500):** a bug in `TaylorStepper.radius`. It converted coefficients to float; a finite mpfr coefficient above 1e308 became `inf`, was read as rho = 0, and froze the step at 2^-400. The threshold is rho < 3e-6 at order 56 and 1.5e-5 at order 64. It is a stall that never recovers, so any reference that reached T is unaffected. Fixed by taking log2 in the run's own arithmetic; values below the float limit go through the same formula as before, so finished results are unchanged. Dry run missed it: E2 refs estimated 0.3 core-h from a sample of 2.
- **Retracted diagnosis:** for about an hour I attributed this to hard binaries admitted by the f64 filter, and told the author so. That was wrong. The variant `nb3_ensemble:r2` (extra 256-bit, order-48 orbit in the filter) was built on it; 10 of its 65 draws were rejected as `truth_max_steps`, which may be the same bug, and 6 of its references stalled on the bug. r2 is superseded and kept on disk. The ensemble now is `nb3_ensemble:r4`: the spec's filter from the base card, same ICs as the base, limit of 150,000 steps, reference 80/600 (check 90/700). Lower protocols missed `tol_check`: #1 at 56/400 (4.5e-30), #2 and #14 at 64/500 (1.9e-30, 2.6e-30). With the fix, all 20 dev ICs reached T = 120 at 64/500 in 2,203–59,407 steps, closest approach 5e-10, so no IC needs excluding on cost. The r2 and r3 card entries and the extra filter stage are removed from the code (commits 76f70d8 and 8787131 hold them); their data is kept and superseded.
- **Invalid references found:** `henon_heiles:E_1_6` failed its check on 18 of 20 ICs (worst 6.8e-6, above the T_pred tolerance 1e-6), `double_pendulum` on 2 of 3 (worst 1.7e-10), `nb3_ensemble` #1 narrowly (4.5e-30). The runner did not act on a failed check, so the E1 dev runs on these two cases started from these references. New variants `henon_heiles:E_1_6_r2` and `double_pendulum:r2` use 96/600 (check 108/700). `Reference` now refuses a failed check and the reference job is marked failed.
- **Old data:** kept on disk, nothing deleted. `experiments/superseded.yaml` lists the replaced cases and `msps.io.iter_runs` skips them.
- **Safeguards:** per-worker memory cap (`--mem-gb`, default 80% of available / workers), reference LRU 8 to 2, queue and status restricted to the current grid. A memory failure inside GMP could abort a worker and hang the pool: check `logs/E*.log` for progress when a stage goes quiet.

## H100 dismissed (E10)
The author said on 2026-11-05 that the `ops` tests can be dismissed and that the tests can run locally. RQ8 is therefore not answered for the H100. History:
`ssh ops` reaches a VMware VM with **no GPU visible**: no `/dev/nvidia*`, no `nvidia-smi`, no driver, `lspci` shows only VMware SVGA (checked 02:34, 2026-11-05). The first night's benchmark did not run; the pip failure in `logs/ops_night_2026-11-05.log` (jax CUDA wheels, network) is secondary. `ops_night.sh` now checks for the GPU first and exits without copying or installing anything if absent. Left on ops: `~/msps1` (306 MB, including a references copy) with a venv without JAX; nothing running.

## Open questions for Alexandre
1. The pilot's `sum()` is compensated on Python ≥ 3.12 (see above). Should the paper quote the plain-IEEE numbers from the new engine? The T_pred table is unaffected.
2. Escape criterion: Stone & Leigh give none that is operational. Is the definition above acceptable, and is r > 10 a the factor you want?
3. Re-synchronise u = 1/r each step, to test whether H3 recovers on nb3 at low n? Method change, not done.
4. Scale Taylor coefficients by h^k, to remove the f32 and dd range failures? Changes the tape, not done.
