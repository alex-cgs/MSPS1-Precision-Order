# MSPS1 experiment digest

T_pred is the time a run stays within tolerance of a validated reference; tolerance is 1e-6 below unless stated. n is the Taylor order. "Censored" means the run was still correct when the simulation ended, so the value is a lower bound.

**Bottom line.** The crossover law holds where it can be tested. Statistics outlive trajectories on Lorenz, Rössler and Hénon–Heiles, but not on the 3-body ensemble at low n. The cause there is largely our polynomialized form of the equations, not the physics. The model fails to transfer across initial conditions (ICs) on that same ensemble. The H100 was not tested.

## What was run
- **Systems:** E0–E11 as planned, covering Lorenz-63, Rössler, Hénon–Heiles, Lorenz-96, FPUT, Duffing, double pendulum, Kepler, 3-body (pilot and ensemble), N-body up to N = 10 and iterated maps.
- **E2 grid:** n = 4 to 32 (even) at 24, 53, 80, 106, 113, 160 and 256 bits, tolerances 1e-6 and 1e-3, 100 ICs per system, one for the pilot (the spec asked for 1,000; I followed AGENT_PROMPT). E11 used 5 ICs per system and E8 one.
- **Hardware:** Ryzen 5 7600X (11 of 12 threads, nice 5) and GTX 1080 Ti.
- **Compute:** 427 core-hours of queued jobs, dev and paper grids (352 on the 88,952 finished, 75 on failed ones), plus E10 at 3.7 h GPU and 3 h on one CPU core. This excludes runs I killed by hand and superseded attempts. Elapsed: 4 Nov 23:57 to 7 Nov 18:04.

## Main findings
**Crossover law (RQ1, H1): supported up to 80 bits, untested above.** T_pred rises with n, then plateaus, and after that only more bits help.

| Median T_pred at n = 8, 16, 24, 32 | Lorenz | Rössler | 3-body ensemble |
|---|---|---|---|
| f64 | 1.07, 19.36, 19.59, 19.59 | 15.81, 234.5, 264.3, 264.3 | 2.47, 40.2, 45.2, 45.2 (20% censored) |
| 113 bits | 1.07, 19.78, 38.59, 57.26 | 15.81, 234.4, 486.7, 733.8 (17%) | 2.47, 51.0, 120, 120 (90%) |

- **Crossover order:** measured and modelled n* agree within about 2.3 orders at 24, 53 and 80 bits. For Lorenz they are 8/8.8, 16/16.8 and 24/24.4.
- **Bits per order:** measured values are Lorenz 3.50, Rössler 3.11 and Hénon–Heiles E = 1/6 3.99, against 3.59, 3.59 and 3.71 predicted and about 3.5 in the pilot. Each rests on 3 points on an even-order grid, so treat them as rough.
- **Above 80 bits:** n* is mostly beyond n = 32 or hidden by censoring.
- **Local error law:** slopes are −1.06 to −1.16 on most systems. ρ/h of about 12–14 follows largely from our step rule, so it is not a property of the systems.
- *Figures: `e2_tpred_tol1e-6`, `e2_crossover`, `e1_local_law`.*

**Model accuracy (RQ2, H2): borderline on four systems, fails on the 3-body ensemble.** Results are out of sample, fitted on 50 ICs and tested on 50.

| System | Two-constant model: rms decades (share within 1 decade) | FTLE variant | T_pred median relative error |
|---|---|---|---|
| Hénon–Heiles E = 1/6 | 1.11 (79%) | 0.91 (79%) | 3.8% |
| Hénon–Heiles E = 1/8 | 0.80 (80%) | 0.91 (76%) | 6.6% |
| Lorenz | 1.12 (63%) | 0.51 (97%) | 10.7% |
| Rössler | 1.17 (62%) | 0.58 (90%) | 16.3% |
| 3-body ensemble | 3.24 (25%) | 2.66 (26%) | 100% |

The FTLE variant brings all four non-3-body systems under one decade, with no extra fitted constants. *Figure: `e3_pred_vs_measured`.*

**Statistics vs trajectories (RQ3, H3): holds on smooth flows, fails on the 3-body ensemble at low n.**
- **Lorenz:** with f32, T_pred is 0.25 (one output-grid step), yet the Lyapunov exponent λ stays within 0.0031 of 0.9049 at every (n, p). W1 is at most 0.0435 against a split-half floor of 0.0326.
- **Rössler:** λ is within 0.0012 of 0.0713. W1 is at most 0.0078 at n ≥ 4 against a floor of 0.0066, and up to 0.0175 at n = 2. Lorenz-96 at N = 8 behaves the same way (20 runs).
- **Hénon–Heiles:** n = 2 is wrong in every arithmetic (λ off by 0.10–0.12, the orbit runs away). At E = 1/8, n = 4 is off by 0.016, which is 68% of λ. At n ≥ 6 the error is at most 0.0085, still up to 37% of λ there, with no trend in n and no floor computed for λ.
- **3-body ensemble:** the escape fraction misses the reference 0.84 at low n, whatever the precision.

| Escape fraction at n = 4, 6, 8, 12, 16, 24, 32 | |
|---|---|
| f64 | 0.10, 0.26, 0.35, 0.66, 0.81, 0.73, 0.72 |
| 113 bits (no n = 32) | 0.12, 0.25, 0.34, 0.64, 0.82, 0.86 |
| f32 | 0.11, 0.34, 0.25, 0.04, 0.01, 0, 0 |

Two mechanisms explain most of this:
- **Lifted variable drift:** at low n, u = 1/r drifts away from the true 1/r (u·r reached 2e8–6e8 with the bodies far apart), the step collapses and the run dies before it can escape. In a replay of 73 such deaths, all had |u·r − 1| > 1.
- **Exponent range:** Taylor coefficients grow like ρ^−k and overflow f32, and dd at very small steps. This causes the f32 fall and probably dd's blow-ups at n = 32 (13 of 100). The f64 dip at n ≥ 24 comes from 7–11 step underflows whose cause was not checked; 113 bits has none at n = 24.

*Figures: `e5_stats_vs_trajectories`, `e5_escapes`.*

**Representation (RQ6): the bit count decides among floats, but rounding to 2^−p differs.**
- f64 and MPFR-53 give identical T_pred, as do f32 and MPFR-24.
- Rationals rounded to 2^−53 beat f64 on Lorenz (23.14 vs 18.86) and lose on the pilot (63.64 vs 70.43).
- Maps barely change at 53 bits (logistic 36, 36, 36 iterations; tent 38, 38, 37).
- *Figure: `e9_representation`.*

**Structure (RQ7, one IC, walls measured under load): mainly invariants change, not T_pred.**
- **Best f64 T_pred on the pilot:** Taylor 70.43, Gauss–Legendre 71.14, DOP853 70.43, Yoshida 78.81, RK4 40.35.
- **Hénon–Heiles E = 1/8:** Yoshida reaches 546, while Taylor, DOP853 and Gauss–Legendre are censored at 800. So Yoshida's lead on the pilot does not generalize.
- **Invariants:** Gauss–Legendre at s ≥ 4 holds pilot energy below 1e-11; the best Taylor run reaches 1e-9.
- *Figure: `e8_cost_vs_accuracy`.*

**Hardware (RQ8; one CPU core vs GTX 1080 Ti only).**
- **Throughput:** at n = 16, Lorenz f64 in steps/s is 1.53e5 vs 1.25e5 at 100 members and 1.21e6 vs 6.49e6 at 10,000, with the GPU reaching 1.52e7 at 100,000. The GPU wins dd at every batch size.
- **Same results across devices:** f64 and dd match bit for bit on every finite value through 3,200 steps (100 members, n = 8, 16, 24). On the 3-body ensemble few members stay finite that long: 9 of 100 at 3,200 steps at worst.
- **f32 does not match:** 71–75 of 100 Lorenz members differ after 3,200 steps, by up to 35. GPU f32 division is not correctly rounded (29.8% of random divisions differ, by up to 2 ulp); f32 add and multiply and all f64 operations match. It is the likely source, but replacing it was not tried.
- *Figures: `e10_throughput`, `e10_cost_accuracy`.*

**Other results.**
- **Spurious dynamics (RQ4):** fake map periods grow with bits (logistic: 40 in f16, 4,344 in f32, 36,077 at 32 bits). Duffing at γ = 0.28 and 0.29 changes regime through precision alone. *Figure: `e6_spurious`.*
- **Chaos indicator (RQ5):** sensitivity to (n, p) tracks FTLE, with Spearman 0.914 and AUC 0.954 on 1,266 points at n = 16. *Figure: `e7_indicator`.*
- **Dimension:** f64 T_pred at n = 32 falls from about 30 at N = 4 (20–100% censored) to 7.9–13.3 at N = 6 (0%; 25.0 and 40% for the hierarchical case), 3.9–6.1 at N = 8 (circle 40%, others 0%) and 3.2–4.2 at N = 10 (0%). f64 n* stays near 16. Lorenz-96 shows no such trend (11.1, 15.1, 11.3 at N = 8, 16, 40; 0%). *Figures: `e11_dimension`, `e11_lorenz96`.*

## Contradictions with the hypotheses or the pilot
- **H2:** fails on the 3-body ensemble. On the other systems the rms sits near one decade, and above it on three of them.
- **H3:** fails on the 3-body ensemble at low n, largely because of a polynomialization artefact.
- **RQ9:** the factor (d−1)n+1 is an upper bound, not a law. N-body reaches 0.86–1.00 of it and Lorenz 0.81–0.93, but Hénon–Heiles only 0.56–0.73. Its Taylor map has degree n/2+1, and growth follows that (7.24 at n = 12, where the law gives 13). The pilot's case holds (32.30 vs 33). *Figure: `e4_exact_growth`.*
- **dd is not a clean 106-bit stand-in at high n:** on one N = 8 run at T = 30, n = 32, dd gives T_pred 4.76 against 24.13 for 113 bits.
- **Pilot:** T_pred is reproduced at every order, and E0 passes 68 of 69 checks. The miss is the fitted c_r, 4.089 against 4.21 in the spec. Rerunning the pilot itself gives 4.197, and its `sum()` is compensated on Python ≥ 3.12.
- **Stone & Leigh:** the paper gives no operational escape test, so the criterion here is my own (positive energy, moving outward, r > 10 a).

## Limitations
- **Censoring:** high-precision medians are lower bounds (113-bit censoring at n = 32: Hénon–Heiles E = 1/8 99%, 3-body ensemble 90%).
- **N = 8, 10:** T = 8 rather than 30, so 113-bit values are capped, and 5 ICs per family. Three ICs needed the order 80/600-bit reference. The only T = 30 data is one IC each, circle family. The amplification run for cluster N = 8 #4 was skipped (about 15 h, over the 6 h limit); no reported number uses it.
- **E5 vs E2:** E5 T_pred uses a coarse grid and its own step rule, so it is not comparable with E2. Hénon–Heiles n = 2 runs have mixed 60 s and 3,600 s caps.
- **Not done:** H100; a 3-body grid in E7; PDE work and confidence regions; Carleman and surrogate models; f80 and Arb in the study grids.
- **References:** the base protocol failed its check on 8 E2 ICs and on N ≥ 8. I raised precision for those and kept the check at 1e-30.
- **Over budget:** E1 (12.4 h vs 2) and E5 (25.9 h vs 20). E11 with E11hi used about 150 core-hours including failures, against a budget of 10 h.

## Suggested next steps
1. **Decide whether to re-synchronize u = 1/r each step,** then rerun E5 on the 3-body ensemble at low n. This is the direct test of whether H3 recovers. It changes the method, so it is your call.
2. **Decide whether to scale coefficients by h^k,** which removes the range failures in f32 and dd but changes the tape.
3. **Test H1 above 80 bits** with n up to about 64 and longer T.
4. **Run E8 on an ensemble,** so the Yoshida and Gauss–Legendre comparison rests on more than one IC.
5. **Run E10 on an H100** if one with a visible GPU becomes available (`ops` showed none). Also canonicalize NaN in the fingerprint: NaN sign alone caused the 3-body hash mismatches.
6. **Answer two open questions in `PROGRESS.md`:** which c_r to quote, and the escape criterion.
