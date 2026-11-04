# MSPS1 — Truncation order, arithmetic precision and chaos

Project specification and working memory for the MSPS1 thesis project (Mathematics and Computational Science).

This document records everything decided so far: the research questions, the mathematics, the pilot results, the test systems, the methods, the experiment plan, the code architecture, the test strategy and the hardware plan. It is written for two readers:

- **the author** (who does the research and writes the paper), and
- **Claude Code running locally in this folder** (who implements, tests and runs code on request).

> **Rule for the coding assistant:** the research questions, definitions and metrics in this document are fixed by the author. Do not change them silently. If something here is wrong or ambiguous, say so and propose a change. Never invent, round up or "smooth" results: every number reported must come from a stored run with its config.

---

## Contents

1. Project summary
2. Research questions and hypotheses
3. Positioning against prior work
4. Mathematical core
5. Pilot run (baseline results)
6. Test systems (case cards)
7. Methods and arithmetic back-ends
8. Experiment plan (studies E0–E13)
9. Software architecture
10. Testing strategy
11. Hardware and compute plan
12. Phased work plan
13. Paper plan
14. Later work: confidence regions
15. Prompting guide for Claude Code
16. Glossary
17. Key references

---

## 1. Project summary

When a computer integrates an ODE, information is lost in two separate ways:

- **truncation**: the method approximates the solution (here mainly a Taylor series cut at order *n*), and
- **rounding**: the arithmetic stores numbers with a finite significand of *p* bits.

In a chaotic system both errors are amplified roughly like $e^{\lambda t}$, so every run eventually follows a different orbit than the true one. The project studies the two error sources **jointly**, across many nonlinear systems, and asks:

- how long a run stays on the true orbit as a function of $(n, p)$,
- whether this can be predicted from measurable quantities,
- where raising *n* stops helping because *p* is the bottleneck (the **crossover order** $n^*(p)$), and
- whether the **statistics** of the dynamics stay correct long after individual **trajectories** are lost.

The core tool is a **single code path** that runs the same algorithm in float32, float64, double-double, multiple-precision floating point (MPFR), ball arithmetic and exact rationals. Every system is first rewritten **symbolically** (SymPy) into a polynomial ODE so that the Taylor method has exact recurrences and exact rational initial data are possible.

Origin of the idea: symbolic computation is exact, but practical computation always truncates (series order) or rounds (`evalf(k)`, floating point). Feeding an approximate state into the next step, again and again, behaves like a chaotic map of the error. The Kafetzis–Moysis–Volos paper showed that Taylor truncation of *sin* in the sine map changes the dynamics. This project moves the question from "the model is truncated" to "the solver and the arithmetic are truncated".

---

## 2. Research questions and hypotheses

**Primary (paper 1).**

- **RQ1 — Crossover law.** For a Taylor method of order *n* in *p*-bit arithmetic, the predictability time $T_{\text{pred}}(n,p)$ grows linearly in *n* until a crossover order $n^*(p)$. Beyond that point only more precision helps.
  - *H1:* $n^*(p) \approx \dfrac{p\log_{10}2 + c}{\log_{10}(\rho/h)}$, with the same functional form across systems. The pilot gives about one order per 3.5 bits.
- **RQ2 — Predictive model.** $T_{\text{pred}}$ and the error $E(t)$ can be predicted from the local truncation error, the rounding unit and the measured amplification $A(t)$, plus at most two fitted constants per system.
  - *H2:* the prediction holds out-of-sample over an ensemble of initial conditions, to within one decade of error.
- **RQ3 — Statistics outlive trajectories (main finding to look for).** Long-run statistics stay correct at much lower $(n,p)$ than trajectories do. Examples: Lyapunov exponent estimates, invariant measures and histograms, escape-time distributions, energy spectra.
  - *H3:* statistical accuracy saturates at a modest $(n,p)$ even when $T_{\text{pred}}$ is tiny. A failure of H3 in a specific system is also a publishable result.

**Secondary (same paper or follow-ups).**

- **RQ4 — Spurious dynamics.** Where in the $(n,p)$ plane does the computed system become qualitatively wrong? Examples: fake periodic orbits, shifted bifurcations, destroyed KAM islands, wrong escape outcomes.
- **RQ5 — Chaos indicator.** Is the sensitivity of a run to $(n,p)$ a usable chaos indicator? Compare it with finite-time Lyapunov exponents (FTLE) or MEGNO.
- **RQ6 — Representation.** Does the representation matter, or only the bit count? Compare float with *p* bits, MPFR with *p* bits and rationals rounded to $2^{-p}$.
- **RQ7 — Structure.** At equal cost, do structure-preserving methods (Gauss–Legendre, Yoshida) change $T_{\text{pred}}$, or only invariants and statistics?
- **RQ8 — Hardware.** How do cost and accuracy trade off on a CPU, a GTX 1080 Ti and an H100? Does the hardware change the bitwise results, and therefore the trajectories?
- **RQ9 — Exact arithmetic.** For a polynomial right-hand side of degree *d*, does the bit size of an exact Taylor step grow by a factor of about $(d-1)n+1$ per step? The pilot gives ×31.6–33.6 against a predicted ×33 for *d* = 5, *n* = 8.

**Later (paper 2).** Calibrated confidence regions after time *t*, built from ensembles (Section 14).

---

## 3. Positioning against prior work

A full list with PDF links is in the Google Drive spreadsheet **MSPS1 / MSPS1 bibliography** (43 entries). Section 17 lists the key ones.

| Prior work | What it already does | What this project adds |
|---|---|---|
| Kafetzis, Moysis, Volos 2023; Moysis et al. 2025 review; Baranwal et al. 2026 | Truncated Taylor polynomials of sin or exp define *new* chaotic maps (model truncation) | Solver and arithmetic truncation applied to the *same* model; a joint $(n,p)$ map |
| Liao: Clean Numerical Simulation (CNS), 2009–2020 | High-order Taylor plus multiple precision, both raised until results converge; includes the three-body problem | The full $(n,p)$ plane, a predictive model, the crossover law, ensembles over initial conditions, statistics vs trajectories |
| Wang & Li 2014 | $T_c = (\ln B/\lambda)K + C$ for *K* digits of precision | The truncation axis, and the interaction between the two axes |
| Kehlet & Logg 2013, 2017 | A posteriori separation of round-off and discretization error; order 200 at 400 digits on Lorenz | Ensembles, many systems, the crossover, statistics |
| Boekholt & Portegies Zwart 2015–2020; Portegies Zwart 2026 | Arbitrary-precision N-body (Brutus); fp16–fp128 and Posits on chaotic N-body | Taylor order as a second controlled axis; a predictive model |
| Klöwer et al. 2023; Lorenz 1989, 2006 | Low precision or step size creates spurious periodicity or chaos | A systematic map of where spurious dynamics starts (RQ4) |
| Shadowing literature (Hammel–Yorke–Grebogi; Sauer et al.; Hayes & Jackson) | When numerical orbits stay close to true orbits | Empirical shadowing and predictability times as functions of $(n,p)$ |

The claim of novelty must stay narrow and defensible:

- the joint $(n,p)$ map,
- the crossover law and the predictive model, validated across systems and initial-condition ensembles,
- the statistics-versus-trajectory result.

---

## 4. Mathematical core

### 4.1 Polynomialization

The Taylor method needs exact recurrences, and exact rational arithmetic needs a polynomial right-hand side (RHS). Each system is rewritten as $\dot z = f(z)$ with $f$ polynomial, by adding auxiliary variables:

- $u_{ij} = 1/r_{ij}$ for gravity: $\dot u_{ij} = -u_{ij}^3\,(\mathbf d_{ij}\cdot\mathbf w_{ij})$, where $\mathbf d = q_i-q_j$ and $\mathbf w = v_i - v_j$.
- $s=\sin\theta$, $c=\cos\theta$ for trigonometric terms: $\dot s = c\dot\theta$, $\dot c = -s\dot\theta$.
- $(C,S) = (\cos\omega t, \sin\omega t)$ for periodic forcing: $\dot C=-\omega S$, $\dot S=\omega C$.
- $e = \exp(g)$: $\dot e = e\,\dot g$.

Every auxiliary variable creates a **constraint** (for example $C_{ij}=u_{ij}^2 r_{ij}^2-1=0$). It must be monitored. For gravity, SymPy shows

$$\dot C_{ij} = -2u_{ij}^2(\mathbf d\cdot\mathbf w)\,C_{ij}.$$

So $C=0$ is invariant, but numerical errors in $C$ are neither damped nor corrected.

**Invariants of the lifted gravity system.** The energy written in lifted variables,

$$E = \tfrac12\sum_i m_i|v_i|^2 - \sum_{i<j} m_i m_j u_{ij},$$

satisfies $\dot E = 0$ exactly, without using the constraint. Linear momentum and angular momentum are also conserved.

### 4.2 Taylor recurrences (Taylor-mode automatic differentiation)

Write $x(t_k+s)=\sum_{m=0}^{n} x_{[m]}s^m$. Then

$$x_{[m+1]} = \frac{f_{[m]}}{m+1},\qquad (ab)_{[m]}=\sum_{l=0}^{m}a_{[l]}b_{[m-l]}\quad\text{(Cauchy product).}$$

The cost per step is $O(n^2)$ times the number of products in *f*. The step is evaluated with Horner's rule at $s=h$.

**Validation rule.** In exact rationals, the recurrence coefficients must equal the Lie-series coefficients $L_f^m x/m!$ computed by SymPy, for all variables and low orders. The pilot passed this for orders 0–4.

**Generalization required.** The engine must **generate the recurrences automatically** from the SymPy polynomial RHS (a "Taylor tape" of `add`, `sub`, `mul`, `scale` and `const` operations). The pilot hand-coded them for three bodies, which does not scale to new systems.

### 4.3 Floating-point model

- The standard model is $\mathrm{fl}(a\circ b)=(a\circ b)(1+\delta)$ with $|\delta|\le u = 2^{-p}$, the unit roundoff for a *p*-bit significand (round to nearest).
- float64 has *p* = 53, float32 has 24, float16 has 11, x87 long double has 64 and IEEE quad has 113. Double-double gives about 106 bits.
- Rounding errors accumulate over $N$ steps roughly like $\sqrt N\,u$ when they behave randomly (Brouwer's law) and like $N u$ when they are biased.

### 4.4 Error model (tested in the pilot)

**Local truncation error.** With step *h* and local convergence radius $\rho$, Cauchy estimates give

$$\varepsilon_{\text{trunc}}(n)\approx K\left(\frac{h}{\rho}\right)^{n+1}\quad\Rightarrow\quad \log_{10}\varepsilon_{\text{trunc}}\approx \log_{10}K-(n+1)\log_{10}\frac{\rho}{h}.$$

**Global error.** Each step's error is carried forward by the linearized flow:

$$E(t)\approx\sum_k \|\Phi(t,t_k)\,\varepsilon_k\|,\qquad \Phi = \text{fundamental matrix of the variational equation}.$$

The working model replaces $\Phi$ by a measured amplification $A(t)$, the growth of a tiny perturbation of the reference:

$$E(t)\approx\Big[10^{c_t}\varepsilon_{\text{trunc}}(n)+10^{c_r}2^{-p}\Big]A(t).$$

The constants $c_t$ and $c_r$ absorb the number of steps and how errors are spread along the orbit. They differ because truncation errors concentrate in hard steps, while rounding errors occur at every step.

**Predictability time.** $T_{\text{pred}}$ is the first $t$ with $E(t)>\text{tol}$. If $A\approx e^{\lambda t}$, then $T_{\text{pred}}\approx\lambda^{-1}\ln(\text{tol}/\varepsilon_{\text{eff}})$. This is linear in *n* (slope $\ln(\rho/h)/\lambda$ per order) and linear in *p* (slope $\ln 2/\lambda$ per bit).

**Crossover order.**

$$n^*(p)=\frac{p\log_{10}2+c_t-c_r+\log_{10}K}{\log_{10}(\rho/h)}-1.$$

**Exact-arithmetic growth.** If the RHS has total degree *d*, a Lie derivative raises the polynomial degree by $d-1$. So $x_{[m]}$ has degree $(d-1)m+1$ in the state, and one step of order *n* multiplies the bit size by about $(d-1)n+1$. That gives $4n+1$ for gravity (*d* = 5) and $n+1$ for Lorenz (*d* = 2). Storage grows like $((d-1)n+1)^k$ after *k* steps.

**Known weakness (from the pilot).** In N-body problems, both the error injection and the amplification happen in bursts at close encounters. One amplification curve $A(t)$ smooths over the timing of the bursts, so the model missed at *n* = 10 and 14. Planned fix: use time-resolved FTLE and inject errors step by step through the tangent map.

### 4.5 Lyapunov tools

- **Largest Lyapunov exponent:** two-trajectory method with renormalization, or tangent-map QR (Benettin) at reference precision.
- **FTLE:** $\lambda_\tau(t) = \tau^{-1}\ln\|\Phi(t+\tau,t)\|$, giving the distribution and time series of local stretching.
- **Variational equations:** also polynomial after polynomialization, so the same Taylor engine can integrate them.

### 4.6 Shadowing

A numerical orbit is a pseudo-orbit with one-step error $\delta$. In hyperbolic regions there is a true orbit within $\varepsilon(\delta)$ of it for a finite time. This justifies treating a run as "a true orbit from a perturbed initial condition", which RQ3 and the confidence-region work rely on. Read the Hayes & Jackson 2005 survey first.

### 4.7 Metric definitions (fixed; do not change without the author)

| Symbol | Definition |
|---|---|
| $E(t)$ | $\max_i \lvert q_i(t)-q_i^{\text{ref}}(t)\rvert_\infty$ over position components. Use relative distances for N-body when stated. |
| $T_{\text{pred}}(\text{tol})$ | First grid time with $E(t)>\text{tol}$. Censored at $T$ if never crossed; record the censoring. Default tol = $10^{-6}$; report $10^{-3}$ as well. |
| $\varepsilon_{\text{loc}}(n)$ | One-step error from reference states, computed in high precision. Report the median and interquartile range over ≥ 60 states. |
| $A(t)$ | Separation of a $10^{-20}$ (or $10^{-30}$) perturbation divided by its size, at reference precision. |
| $\Delta H(t)$ | $\lvert H(t)-H(0)\rvert$ for each conserved quantity. |
| $\Delta C(t)$ | $\max\lvert\text{constraint}\rvert$ (for example $u^2r^2-1$). |
| Cost | Wall time, steps, and operations per step. For exact arithmetic, also the bit size per number. |
| Statistical error | Distance between statistics of a run (or ensemble) and the reference ensemble. Examples: the λ estimate, the Wasserstein-1 distance between histograms, the escape-time distribution (Kolmogorov–Smirnov test). |

A **reference** is a run whose own error is shown to be far below every error measured against it. The standard check is a second run at a higher order and precision on the same time grid, with the difference reported.

---

## 5. Pilot run (baseline results)

**Setup.**

- Planar three-body problem, equal masses, *G* = 1.
- Positions on a 3-4-5 triangle: $q_1=(1,3)$, $q_2=(-2,-1)$, $q_3=(1,-1)$.
- Velocities $v_1=(0,-\tfrac14)$, $v_2=(\tfrac14,\tfrac14)$, $v_3=(-\tfrac14,0)$; total energy $E=-79/120$.
- All initial data are rational, including $u_{ij}(0)=\tfrac15,\tfrac14,\tfrac13$.
- Integration window $T=120$.

**Design.**

- A shared dyadic step schedule $h=2^{\lfloor\log_2(\rho/8)\rfloor}$, capped at 1/2, built once by the reference run: 2,605 steps with *h* from $4.8\times10^{-7}$ to 0.5.
- Reference: order 56 at 400 bits, checked against order 64 at 500 bits. The maximum difference was $1.7\times10^{-41}$.
- Sweep: $n\in\{4,6,\dots,20,24,28\}$ × {float64, 80, 113, 160 bits}, plus exact rationals at *n* = 8.

**Results (regression targets for the new code on the same setup).**

| Quantity | Value |
|---|---|
| Closest approach | $4.1\times10^{-4}$ at $t\approx87.4$; other close approaches at $t\approx40, 79$ |
| Amplification over $[0,120]$ | $7.8\times10^8$; mean exponent $\lambda\approx0.16$ |
| Local law | $\log_{10}\varepsilon = -2.29 - 1.07(n+1)$, so $\rho/h\approx11.7$ (designed range 8–16) |
| Fitted constants | $c_t = 7.43$, $c_r = 4.21$; root-mean-square misfit 0.60 decades over 6,207 points |
| Crossover $n^*$ | float64: 14.8; 80 bits: 22.5; 113 bits: 31.8; 160 bits: 45.0 |
| $T_{\text{pred}}$ (tol $10^{-6}$), float64 | n = 4: 0.5; 6: 2.4; 8: 2.7; 10: 11.3; 12: 31.3; 14: 57.5; 16 and above: 70.4 |
| $T_{\text{pred}}$, 80–160 bits | Same up to n = 12; 14: 57.6; 16: 79.1; 18: 118.8; 20 and above: > 120 |
| Model $T_{\text{pred}}$ | 10: 4.3; 12: 27.2; 14: 40.4; 16: 79.0; 18: 118.8. Misses at n = 10 and 14 |
| Exact rationals, n = 8 | Bits per number 224 → 7,521 → 238,496 → 7,541,120 (×33.6, ×31.7, ×31.6; predicted ×33). 4 steps took 175 s and reached t = 0.875 |
| Invariants vs error | 80 bits, n = 16: energy error $7\times10^{-14}$ but position error $5\times10^{-5}$ at T |
| Shared-schedule failure | Runs that have already left the orbit blow up at the next close encounter, because the shared steps no longer fit their trajectory |

The pilot scripts are in `pilot/` (see `pilot/README.md`). Treat them as a **reference snapshot**, not as the architecture to extend.

---

## 6. Test systems (case cards)

**Are formulas enough?** No. Each system needs a full **case card**, because the formulas alone leave too much undecided. A case card is a small YAML file plus a Python module, containing:

- the equations (original form) and the polynomialized form (generated by SymPy, not typed by hand),
- parameters and the regime they produce (regular, chaotic, mixed), with a source for any quoted value,
- initial-condition generators (deterministic, seeded, rational where possible),
- conserved quantities and constraints,
- a reference protocol (order, precision, step rule, check run),
- the horizon *T*, tolerances and observables (what $E(t)$ measures for this system),
- known values to test against (oracles), and
- the role of the system in the study.

The **experiment matrix** (Section 8) then combines case cards × methods × $(n,p)$ grids × initial-condition ensembles.

Values marked *verify* are from memory or the literature and must be checked before they are used as test targets.

### 6.1 N-body family (main line)

**Polynomialized form:** $q, v$ per body plus $u_{ij}=1/r_{ij}$ per pair. That is $2dN+N(N-1)/2$ variables in dimension *d*, with an RHS of degree 5. For N = 10 in the plane: 40 + 45 = 85 variables.

| Case | Purpose | Initial data |
|---|---|---|
| **NB2-Kepler** | **Exact oracle.** The two-body problem has a closed-form solution (Kepler's equation), so the true error is known without a reference run | Eccentricities 0, 0.5, 0.9; solve Kepler's equation with mpmath at 200+ digits for the oracle |
| **NB3-pilot** | Regression against the pilot | 3-4-5 triangle with quarter velocities (Section 5) |
| **NB3-Pythagorean** | Classic chaotic scattering with very close encounters and an eventual escape | Masses 3, 4, 5 at rest at $(1,3), (-2,-1), (1,-1)$ |
| **NB3-figure-eight** | Regular (periodic, stable) orbit: tests behaviour when $\lambda\approx0$ | Chenciner–Montgomery orbit, equal masses: $x_1=-x_2=(0.97000436,-0.24308753)$, $x_3=0$, $v_3=(-0.93240737,-0.86473146)$, $v_1=v_2=-v_3/2$ (*verify digits*) |
| **NB3-Lagrange** | Equilateral relative equilibrium, unstable for equal masses: instability from rounding alone | Equilateral triangle with rotating velocities |
| **NB3-ensemble** | Statistics over initial conditions (RQ2, RQ3) | 100–1,000 rational initial conditions from a seeded generator, bound energy, zero momentum, filtered for a minimum distance and no early escape |
| **NBN-circle** | N = 4…10, exact rational distances | See the construction below |
| **NBN-alternating** | N = 4…10: two concentric rings with alternating masses or radii; a hierarchical binary plus perturbers | Rational coordinates; $u_{ij}(0)$ computed at the target precision when not rational |
| **NBN-cluster** | Plummer-like random cluster with virial-scaled velocities | Seeded; float-generated, then converted to exact rationals |

**Exact rational distances for any N.** Pick rationals $t_k$ and set $\cos\alpha_k=\frac{1-t_k^2}{1+t_k^2}$, $\sin\alpha_k=\frac{2t_k}{1+t_k^2}$. Place body *k* at $P_k=R(\cos2\alpha_k,\sin2\alpha_k)$. Then

$$|P_j-P_k| = 2R\,|\sin(\alpha_j-\alpha_k)| = 2R\,|\sin\alpha_j\cos\alpha_k-\cos\alpha_j\sin\alpha_k|,$$

which is rational. So all positions **and** all $u_{ij}(0)$ are exact rationals for any N. Choose the $t_k$ so the bodies are roughly evenly spread, add rational velocities, and set the total momentum to zero.

**Close encounters.** Do not hide them. Record the minimum pair distance per run. A second step-size mode (each run choosing its own adaptive step) must exist, because the shared schedule fails once a run leaves the orbit (Section 5). Regularization (Levi-Civita, Kustaanheimo–Stiefel, Sundman time transformation) is optional, later work.

**Observables:**

- absolute positions,
- pair distances (more stable under rotation),
- energy, momentum, angular momentum and constraint drift,
- escape events: which body escapes and when (energy-based criterion, *verify* the definition in Stone & Leigh 2019).

### 6.2 Other ODE systems

| Case | Equations (original) | Polynomial? | Regime and parameters | Oracle or known value | Role |
|---|---|---|---|---|---|
| **Lorenz-63** | $\dot x=\sigma(y-x)$, $\dot y=x(\rho-z)-y$, $\dot z=xy-\beta z$ | Yes, *d* = 2 | σ = 10, ρ = 28, β = 8/3; also sweep ρ through the onset of chaos | Largest exponent λ ≈ 0.906 (*verify*); attractor statistics | Clean dissipative baseline; Kehlet–Logg and Liao comparisons |
| **Rössler** | $\dot x=-y-z$, $\dot y=x+ay$, $\dot z=b+z(x-c)$ | Yes, *d* = 2 | a = b = 0.2, c = 5.7 | λ ≈ 0.07 (*verify*) | Weak chaos |
| **Hénon–Heiles** | $H=\tfrac12(p_x^2+p_y^2)+\tfrac12(x^2+y^2)+x^2y-\tfrac13y^3$ | Yes, *d* = 2 | Energies 1/12 (onset), 1/8, 1/6 (escape) | Energy; Poincaré sections | Mixed phase space (KAM islands): RQ4, RQ5 |
| **Double pendulum** | Standard Lagrangian with $\sin,\cos$ | After adding $s,c$ variables | m = l = 1, g = 9.81, large angles | Energy | Mixed phase space; also Kafetzis-style $T_n(\sin)$ model truncation |
| **Forced Duffing** | $\ddot x+\delta\dot x-x+x^3=\gamma\cos\omega t$ | After adding the forcing oscillator | δ = 0.3, ω = 1.2, γ swept (*verify* chaotic values) | Bifurcation diagram | Does truncation shift bifurcations? (RQ4) |
| **Lorenz-96** | $\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F$ | Yes, *d* = 2 | F = 8, N = 8…40 | λ grows with N | Dimension scaling |
| **FPUT-β chain** | $\ddot q_i=(q_{i+1}-2q_i+q_{i-1})+\beta[(q_{i+1}-q_i)^3-(q_i-q_{i-1})^3]$ | Yes, *d* = 3 | N = 16–64 | Energy | Long-time Hamiltonian; structure (RQ7) |
| **Sine map and Taylor maps** | $x_{k+1}=\mu\sin(\pi x_k)$ and $\mu T_w(\pi x_k)$ | Already polynomial for $T_w$ | Kafetzis parameters | Reproduce their λ diagrams qualitatively | Connects to the original idea; map-level RQ4 and RQ6 |
| **Logistic and tent maps** | $x\mapsto rx(1-x)$ | Yes | r = 4; exact-rational orbits | Angelidis et al. 2025 horizon formula | Cheapest test of precision and representation effects |

### 6.3 PDE systems (later; DOLFINx or spectral)

| Case | Notes |
|---|---|
| **Viscous Burgers** (periodic) | Exact solution via the Cole–Hopf transform: an oracle with no chaos. Spectral or finite-element versions. |
| **Kuramoto–Sivashinsky** | Domain length L = 22 for the chaotic regime (*verify*); the ETDRK4 benchmark of Kassam & Trefethen. Truncation knob: number of modes *N*. Statistics: energy spectrum. |
| **Cahn–Hilliard** | An existing DOLFINx demo: mixed P1 elements, θ-method with θ = 0.5, Newton solver. Knobs: mesh size *h*, degree, time step, float32 vs float64. |

**DOLFINx facts (checked in v0.11 docs).**

- Scalar types are float32, float64, complex64 and complex128 (see the "different scalar types" demo, which uses SciPy solvers on a single MPI rank).
- There is **no arbitrary precision**, so PDE work covers only the float32 vs float64 axis plus the discretization knobs.
- UFL is a symbolic form language compiled to numeric kernels by FFCx. It is worth a paragraph in the paper because of the symbolic-programming theme.

---

## 7. Methods and arithmetic back-ends

### 7.1 Integrators (engines)

| ID | Method | Accuracy knobs | Notes |
|---|---|---|---|
| T-n | **Taylor series of order n** (main) | n, h or step rule | Recurrences generated from the SymPy RHS; same code for every number type |
| RK4, DOP853 | Explicit Runge–Kutta | h, tolerance | Baselines |
| GL-s | Gauss–Legendre implicit RK (order 2s) | s, h, tolerance of the fixed-point/Newton solve | Symplectic; the solver tolerance is a second precision knob (RQ7) |
| VER, YOS-k | Störmer–Verlet; Yoshida compositions of order 4, 6, 8 | h, order | Separable Hamiltonians only. Yoshida coefficients are algebraic numbers, e.g. $2^{1/3}$, not rational |
| CARL-N | Carleman linearization truncated at order N | N | Low-dimensional polynomial systems; expected to fail for Lorenz at ρ = 28 (negative control) |
| SUR-n | Taylor surrogate of the *model* (Kafetzis style) | n | Model truncation, compared against solver truncation |

**Step-size rules (separate module):**

- **shared schedule:** built once by the reference run and reused by every run;
- **per-run adaptive:** Jorba–Zou-style rule from the decay of the coefficients;
- **fixed h.**

Each study states which rule it uses.

### 7.2 Arithmetic back-ends (one interface)

| ID | Representation | Library | Significand bits |
|---|---|---|---|
| f16 | IEEE half | NumPy | 11 (maps only; too coarse for ODEs) |
| f32 | IEEE single | NumPy, CuPy, JAX | 24 |
| f64 | IEEE double | Python float, NumPy, CuPy, JAX | 53 |
| f80 | x87 long double | `numpy.longdouble` (platform dependent) | 64 |
| dd | Double-double, via error-free transformations (TwoSum, TwoProd with FMA) | Own implementation; GPU-capable | ≈106 |
| mpfr-p | MPFR | `gmpy2` (fast); `mpmath` (simpler, slower) | any p |
| arb-p | Ball arithmetic (midpoint and radius) | `python-flint` | any p, plus a certified radius |
| q | Exact rationals | `python-flint` `fmpq` | exact |
| qr-p | Rationals rounded to denominator $2^p$ after each step | `fmpq` plus rounding | p |

The interface for every back-end provides: `convert(Fraction)`, `to_float`, `unit_roundoff`, `name`, `bits`, and vectorized versions of add, mul and Horner where the back-end supports them.

**Determinism.** Set the MPFR precision per process, never globally in a shared process. Record the FMA, BLAS and thread settings, and the GPU model and driver.

---

## 8. Experiment plan (studies E0–E13)

Each study is one config file in `experiments/`, writes to `results/raw/<study>/<run_id>/`, and has one analysis script and one figure script.

| ID | Study | Systems | Grid | Main outputs |
|---|---|---|---|---|
| E0 | Validation suite (also in CI) | All | Small | Pass/fail tables |
| E1 | Local truncation law | All ODEs | n = 2…40 at 400 bits | Slope, $\rho/h$, *K* per system |
| E2 | $(n,p)$ sweep | NB3-pilot, NB3-ensemble, Lorenz, Hénon–Heiles, Rössler, NBN | n = 4…32 × {f32, f64, dd, 80, 113, 160, 256} × initial conditions | $T_{\text{pred}}$, $E(T)$, crossover $n^*(p)$ with confidence intervals across initial conditions |
| E3 | Model fit and out-of-sample prediction | As E2 | Fit on half the initial conditions, test on the other half | Misfit distribution; FTLE-based model vs the one-curve model |
| E4 | Exact-arithmetic growth | NB2, NB3, Lorenz, Hénon–Heiles, maps | n = 2…12, few steps | Growth factor vs $(d-1)n+1$ |
| E5 | **Statistics vs trajectories (RQ3)** | Lorenz, Hénon–Heiles, NB3-ensemble, Lorenz-96, KS | $(n,p)$ grid × long runs | λ estimates, histogram distances, escape-time distributions vs $(n,p)$ |
| E6 | Spurious dynamics (RQ4) | Maps, Lorenz, Duffing, Hénon–Heiles | Low p (f16, f32), low n | Periodicity detection, bifurcation shifts, island survival |
| E7 | Chaos indicator (RQ5) | Hénon–Heiles initial-condition grid, NB3 grid | Two or three $(n,p)$ pairs per point | Sensitivity map vs FTLE or MEGNO map |
| E8 | Method families at equal cost (RQ7) | NB3, Hénon–Heiles, FPUT, Lorenz | T-n, GL-s, YOS-k, RK4, DOP853 | $T_{\text{pred}}$, invariants, statistics vs wall time |
| E9 | Representation (RQ6) | Maps, Lorenz, NB3 | f64 vs mpfr-53 vs qr-53, and so on | Whether same-bit-count arithmetics agree |
| E10 | Hardware (RQ8) | NB3-ensemble, Lorenz ensembles | CPU, GTX 1080 Ti, H100 × f32, f64, dd × ensemble size | Throughput, cost per accurate time unit, bitwise reproducibility |
| E11 | Dimension scaling | NBN with N = 2…10, Lorenz-96 | n, p, N | $n^*$ and $T_{\text{pred}}$ vs N |
| E12 | PDE extension | Burgers, KS, Cahn–Hilliard | Mesh or modes × f32/f64 × time step | Same metrics, statistics |
| E13 | Confidence regions (paper 2) | NB3, NBN, Lorenz | Ensembles | Calibrated regions (Section 14) |

**Initial-condition ensembles.** Fix the seeds and store the exact rational initial conditions with the run. Fixed defaults: 100 initial conditions for development, 1,000 for the paper.

**Analysis rules.**

- Report the distribution (median and interquartile range, or quantiles) over initial conditions, not just a mean.
- Treat censored $T_{\text{pred}}$ values properly (survival-style plots, or report the censoring fraction).
- Write down every fitted constant together with the data it was fitted on.

---

## 9. Software architecture

```
msps1/
├─ CLAUDE.md                     # short instructions for Claude Code (read every session)
├─ docs/PROJECT_SPEC.md          # this file
├─ pyproject.toml                # package "msps"; Python ≥ 3.11
├─ environment.yml               # conda env: numpy scipy sympy mpmath gmpy2 python-flint matplotlib pandas pyarrow pytest hypothesis
├─ environment-gpu.yml           # + jax[cuda] or cupy (separate env per machine)
├─ environment-fenics.yml        # DOLFINx from conda-forge (separate env; or Docker)
├─ src/msps/
│  ├─ arith/          # back-ends behind one interface (Section 7.2)
│  ├─ symbolic/       # SymPy: model → polynomialized RHS, invariants, constraints, Lie-series checks, Taylor-tape generator
│  ├─ systems/        # case cards: nbody.py, lorenz.py, henon_heiles.py, … + initial-condition generators
│  ├─ integrators/    # taylor.py, rk.py, gauss_legendre.py, symplectic.py, carleman.py; one Stepper interface
│  ├─ schedules/      # shared, adaptive (Jorba–Zou), fixed
│  ├─ gpu/            # batched float32/float64/double-double Taylor kernels (JAX or CuPy); same algorithm as CPU
│  ├─ diagnostics/    # E(t), T_pred, invariants, constraints, FTLE, Lyapunov, statistics distances
│  ├─ models/         # error model, crossover, fits
│  ├─ runner/         # job queue, checkpointing, resume, provenance, time-window guard
│  └─ io/             # result schema (Parquet + JSON metadata), loading helpers
├─ cases/             # YAML case cards (parameters, initial conditions, tolerances, oracles)
├─ experiments/       # e00_validation.yaml … e13_confidence.yaml
├─ results/
│  ├─ references/     # cached validated references, keyed by (case, ic_id, T, protocol hash)
│  ├─ raw/<study>/<run_id>/   # data.parquet, meta.json (config, git hash, versions, host, hardware, timings)
│  └─ derived/        # tables built from raw
├─ analysis/          # scripts that read results/ and write results/derived/
├─ figures/           # plotting scripts: read only results/, write figures/out/ (light + dark)
├─ bench/             # hardware benchmarks (E10)
├─ pde/               # DOLFINx and spectral PDE cases
├─ tests/             # pytest (Section 10)
├─ pilot/             # reference snapshot of the pilot run (do not extend; port ideas to src/)
└─ paper/             # LaTeX; references.bib from the Drive spreadsheet
```

**Interfaces (sketch).**

```python
class Arithmetic:            # one per back-end
    name: str; bits: int | None
    def convert(self, q: Fraction): ...
    def to_float(self, x) -> float: ...
    def unit_roundoff(self) -> float | None: ...

class System:                # one per case card
    name: str; dim: int; degree: int
    def rhs_sympy(self) -> dict: ...         # polynomial RHS (symbols -> expressions)
    def tape(self) -> TaylorTape: ...        # generated from rhs_sympy, cached
    def initial_conditions(self, ic_id: int) -> list[Fraction]: ...
    def invariants(self, z) -> dict: ...
    def constraints(self, z) -> dict: ...
    def observables(self, z) -> dict: ...

class Stepper:
    def step(self, z, h, arith) -> z_new: ...

class Schedule:
    def next_h(self, t, z, coeffs) -> Fraction: ...
```

**Rules.**

1. **No parameters in code.** Everything comes from YAML, and the resolved config is saved with each run.
2. **Provenance on every run:** git commit (and dirty flag), package versions, host, CPU and GPU model, number of threads, seed, start and end time.
3. **Simulation and plotting never mix.** Figures are produced only from `results/`.
4. **References are cached and validated once,** and reused by every study.
5. **Long jobs are resumable:** checkpoint every few minutes, make the job queue idempotent, and skip runs whose result already exists.
6. **Raw data:** store per-step data only for selected runs. For sweeps, store summaries plus $E(t)$ on a coarse time grid.
7. **Ask the author before deleting or overwriting anything under `results/`.**

---

## 10. Testing strategy

Two separate layers:

- **pytest tests:** fast and deterministic. They run on every change, in CI or before a commit.
- **campaigns:** long experiment runs (hours to days) launched by the runner. These are not tests.

### 10.1 Unit and property tests (fast)

- **Recurrences vs Lie series** in exact rationals, for every system, orders 0–4 (and higher for small systems).
- **Generated tape vs hand-written recurrences** (the pilot) for NB3: exact equality.
- **Convergence order:** halving *h* reduces the one-step error by $2^{n+1}$, and the global error over a smooth interval by about $2^n$ (fit the slope on 4–5 values of *h*).
- **Back-end consistency:** at high precision all back-ends agree to their rounding level; `qr-p` and `mpfr-p` agree statistically.
- **Invariants:** for a short horizon, energy, momentum and angular momentum drift stays below a bound that scales with $u$ and $\varepsilon_{\text{loc}}$.
- **Constraint check:** $u^2r^2-1$ stays at rounding level at the start.
- **Symmetry:** permuting the bodies permutes the solution; rotating the initial conditions rotates the solution; reversing time returns to the start for short horizons.
- **Property-based tests (hypothesis):** random small rational initial conditions, random *n* and arithmetic: no exception, finite results, invariants within bounds.

### 10.2 Physics oracles

- **Two-body (Kepler):** compare with the Kepler-equation solution at e = 0, 0.5, 0.9. The error must follow the local law with no reference run needed.
- **Harmonic oscillator** (linear, exact solution): the Taylor step must be exact up to the truncated tail.
- **Figure-eight:** returns close to its start after one period (*verify* the period value) for high $(n,p)$.
- **Burgers** (Cole–Hopf) for the PDE path.
- **Lorenz:** largest Lyapunov exponent within a tolerance of the literature value (*verify* the value first).

### 10.3 Generated test matrix (extensive; can run for days)

Use `pytest.mark.parametrize`, or a generator script that writes campaign configs, over:

- N-body: N ∈ {2, 3, 4, 5, 6, 8, 10}, configurations ∈ {circle-rational, alternating rings, hierarchical, cluster}, seeds 0…k;
- every ODE case card with its parameter sweeps (Lorenz ρ, Hénon–Heiles energy, Duffing γ);
- methods T-n (n ∈ {2…40}), RK4, DOP853, GL-s (s = 1…6), Yoshida (4, 6, 8);
- arithmetic: all back-ends that apply.

Mark heavy cases with `@pytest.mark.slow` and `@pytest.mark.campaign` so the fast suite stays under a few minutes.

### 10.4 Regression tests against the pilot

The new code on the pilot setup (same initial conditions, schedule rule and reference protocol) must reproduce Section 5 within tolerance:

- local slope −1.07 ± 0.03,
- float64 $T_{\text{pred}}$ plateau at 70.4,
- 80-bit $T_{\text{pred}}$ at n = 16 equal to 79.1,
- exact growth factors near 33.

---

## 11. Hardware and compute plan

**Is a GPU helpful?** Yes, but not for everything.

| Workload | Best hardware | Why |
|---|---|---|
| Multiple-precision (MPFR, Arb) and exact rationals | CPU (many cores) | These libraries run on the CPU; parallelize over runs with `multiprocessing` |
| float32/float64 ensembles (hundreds to thousands of initial conditions) | GPU | The Taylor recurrence is sequential in order *k*, but independent trajectories batch perfectly |
| Double-double (≈106 bits) ensembles | GPU (especially H100) | Built from float64 operations plus FMA, so it runs well where float64 is fast |
| Reference runs | CPU | High precision; done once and cached |

**Hardware facts to quote carefully (verify against vendor specifications before publishing).**

- **GTX 1080 Ti (Pascal):** float64 throughput is about 1/32 of float32 (roughly 11 TFLOPS FP32 vs 0.35 TFLOPS FP64); 11 GB memory.
- **H100:** float64 is about half of float32, with tens of TFLOPS FP64 (higher on SXM with tensor cores); 80 GB memory.

So on the 1080 Ti, **choosing float64 or double-double is expensive**, while on the H100 it is cheap. The cost of precision depends on the hardware, which is a finding for E10.

**Reproducibility across hardware.** GPUs may contract operations into FMA and reorder reductions, so bitwise results differ from the CPU. Because the system is chaotic, these differences grow. Record and study them (compare Qin & Liao 2017, where the number of CPU cores changed the result) instead of suppressing them. Every GPU kernel must have a CPU reference implementation of the same algorithm.

**Machines.**

- **Main machine:** Debian, GTX 1080 Ti. Can run unattended for days.
- **H100:** reached through `ssh ops`, usable only from **02:30 to 04:30**.
  - Jobs are launched by `cron` or `at` at 02:30, with a hard stop at **04:25**.
  - The runner checkpoints every few minutes and refuses to start a job it cannot finish (or checkpoint) before the deadline.
  - Results are synced back with `rsync` after each job.
  - Never leave processes running after 04:30.
  - Never assume the H100 environment matches the local one: record versions in `meta.json`.

**Benchmark protocol (E10).**

- Fixed workloads: NB3 and Lorenz ensembles of size $10^0$ to $10^5$, n ∈ {8, 16, 24}, f32, f64 and dd.
- Measure wall time with warm-up excluded, at least 5 repeats (report the median and spread), throughput in trajectory-steps per second, and memory.
- Also measure the accuracy reached ($T_{\text{pred}}$), so cost can be expressed per accurate time unit.
- Record CPU, GPU and driver models, power mode if available, and library versions.

---

## 12. Phased work plan

Each phase ends with something the author can check. Acceptance criteria are in brackets.

0. **Setup.** Environments, package skeleton, `pytest` running, the pilot reproduced from `pilot/` [the pilot numbers in Section 5 reappear].
1. **Symbolic layer.** SymPy case cards for NB-N, Lorenz and Hénon–Heiles; polynomialization; Taylor-tape generator; Lie-series tests [exact equality for orders 0–4 on all three].
2. **Arithmetic layer.** All CPU back-ends behind one interface; consistency tests [all pass].
3. **Taylor engine plus schedules.** Shared and adaptive step rules; checkpointing runner; provenance [pilot regression within tolerance; two-body oracle passes].
4. **Diagnostics and references.** $E(t)$, $T_{\text{pred}}$, invariants, FTLE, reference cache and validation [reference check below $10^{-30}$ for NB3 and Lorenz].
5. **Studies E1, E2, E4** on NB3-ensemble and Lorenz [crossover with intervals over initial conditions].
6. **E3 model and E5 statistics** [first answer to RQ3].
7. **More systems** (Hénon–Heiles, Rössler, NBN, Lorenz-96, FPUT, maps) and **E6–E9**.
8. **GPU back-ends and E10** (1080 Ti, then H100 in night windows).
9. **E11 and the PDE extension E12** (optional for paper 1).
10. **Writing.** Figures frozen from `results/`; paper drafted by the author.

Fix the metric definitions (Section 4.7) before Phase 5, and write down the predictions for E2 and E5 before running them.

---

## 13. Paper plan

**Working-title ideas** (not final):

- *Truncation order versus arithmetic precision in chaotic integration: a crossover law*
- *When does a higher order stop helping? Taylor integration of chaotic N-body dynamics across precisions*
- *Predictability time as a function of Taylor order and floating-point precision*

**Abstract sketch.**

> Numerical solutions of chaotic systems lose accuracy through two channels, truncation of the method and rounding in the arithmetic, which are amplified exponentially but usually studied separately. We study them jointly for Taylor-series integration of order n in p-bit arithmetic, on the gravitational N-body problem and [other systems]. Symbolic polynomialization lets one integrator run unchanged in IEEE, double-double, multiple-precision and exact rational arithmetic from exact initial data. Over ensembles of initial conditions we measure the predictability time T(n,p), the time a run stays within a tolerance of a validated reference. The local error falls geometrically with n. T grows linearly in n up to a precision-dependent crossover order n\*(p) ≈ [a·p + b], beyond which only precision helps. A model built from measured local errors and finite-time Lyapunov amplification predicts T to within [X]. Long-run statistics remain accurate far beyond T [result of RQ3], while exact arithmetic is infeasible because storage grows like ((d−1)n+1)^k. The results give a practical rule for matching method order to arithmetic precision.

**Structure.**

1. Introduction (gap; 3–4 contributions)
2. Related work
3. Setup and background (systems, polynomialization, invariants, Taylor method, floating-point model)
4. Error analysis (local bound via Cauchy estimates; global error via variational equations and FTLE; $T_{\text{pred}}$; $n^*(p)$; exact growth)
5. Implementation (single code path, back-ends, schedules, references)
6. Experimental design (systems, initial-condition ensembles, grids, metrics, reproducibility)
7. Results (local law; $T(n,p)$; crossover; model check; statistics vs trajectories; invariants; cost and hardware; structure)
8. Discussion (limits: close encounters, bursty amplification; practical rule)
9. Conclusion; appendices (derivations, validation, reproducibility)

**Concepts the author should review:**

- Taylor's theorem, radius of convergence and Cauchy estimates
- Taylor-mode automatic differentiation and Lie series
- quadratization and polynomialization of ODEs
- floating-point error analysis (Higham's book) and Brouwer's law
- the variational equation, Gronwall's inequality, Lyapunov exponents, FTLE and Oseledets' theorem (briefly)
- the shadowing lemma and hyperbolicity
- first integrals and Noether's theorem
- invariant manifolds and constraint stability
- heights of rationals and degree growth under composition
- optional: symplectic integrators and backward error analysis; Levi-Civita and Sundman regularization; Galerkin projection

---

## 14. Later work: confidence regions (paper 2)

**Idea.** Model numerical error as an effective initial uncertainty $\varepsilon(n,p)$; shadowing makes this legitimate. Propagate an ensemble of initial conditions and report regions that contain 95% of the ensemble after time *t*.

**Three regimes.**

1. **Linear:** covariance $\Sigma(t)=\Phi\Sigma_0\Phi^\top$ from the variational equations, giving ellipsoids.
2. **Nonlinear:** Monte Carlo ensembles with quantile or density regions, or polynomial chaos.
3. **Saturated:** the regions converge to the invariant measure, or to the energy shell, or (for N-body) become outcome probabilities such as which body escapes.

**Model.** $\log W(t;n,p)\approx\log\varepsilon(n,p)+\lambda t$ until saturation. Use relative quantities for N-body (pair distances, binary membership).

**Key validation.** Coverage: across many initial conditions, the reference must fall inside the predicted 95% region about 95% of the time.

- If coverage holds, cheap runs give trustworthy uncertainty (RQ3, stated quantitatively).
- If it fails, that marks spurious dynamics (RQ4).
- The rate at which the ensemble spreads is the chaos indicator (RQ5).

---

## 15. Prompting guide for Claude Code

**Start of every session.** "Read `CLAUDE.md` and the relevant section of `docs/PROJECT_SPEC.md` before changing code."

**Good prompt shape:**

- the goal, as a phase or study ID,
- the section of this document to follow,
- the acceptance criterion,
- what not to touch,
- how to report back (which numbers, tables or figures).

**Examples.**

- *Phase 1:* "Implement `msps.symbolic` per spec §4.1–4.2: SymPy polynomialization for the N-body case card (any N, planar and 3D) and a Taylor-tape generator. Add tests that compare tape coefficients with Lie-series coefficients in exact rationals for N = 2, 3, 4 at orders 0–4. Do not modify `pilot/`. Report the test results and the number of tape operations per order."
- *Phase 3:* "Port the pilot to the new engine and reproduce spec §5 on the pilot setup. Write `experiments/e00_pilot_regression.yaml` and a report comparing every number in §5 with the new value."
- *Campaign:* "Generate campaign configs for E2 on NB3-ensemble with 100 initial conditions (seeds 0–99), n ∈ {4…32 step 2}, arith ∈ {f64, dd, mpfr-80, mpfr-113, mpfr-160}, tol = 1e-6 and 1e-3. Estimate the total run time from a 2-configuration dry run before launching. Launch it resumable with nohup, and tell me how to monitor it."
- *H100:* "Write `runner` support for a time window: start at 02:30 via cron on `ops`, never start a job that cannot finish or checkpoint before 04:25, rsync results to the main machine afterwards. Test the window logic locally with a fake clock."
- *Analysis:* "From `results/raw/E2/*`, compute $n^*(p)$ per initial condition by fitting the model in §4.4. Report median and interquartile range per p. Write the table to `results/derived/` and a figure to `figures/out/`. Don't refit anything silently: log the fit inputs."

**Things to always ask Claude Code to do:**

- run the fast test suite before reporting a task as done,
- give an estimated run time before any long campaign,
- state which results are new and which come from the cache,
- flag any result that contradicts Section 5 or a hypothesis, instead of "fixing" it.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Truncation order *n* | Highest Taylor coefficient kept per step |
| Precision *p* | Significand bits of the arithmetic |
| $\rho$ | Local radius of convergence of the Taylor series |
| $T_{\text{pred}}$ | Predictability time: first time the error exceeds the tolerance |
| $n^*(p)$ | Crossover order: beyond it, raising *n* gives no gain at precision *p* |
| $A(t)$ | Measured amplification of a tiny perturbation |
| FTLE | Finite-time Lyapunov exponent |
| CNS | Clean Numerical Simulation (Liao) |
| Polynomialization | Adding variables so the RHS becomes polynomial |
| Shared schedule | One list of step sizes, made by the reference run and reused by all runs |
| Reference | A validated high-order, high-precision run used as the truth |
| Censored | The run never crossed the tolerance before *T* |

---

## 17. Key references (full list with links in the Drive spreadsheet)

- I. Kafetzis, L. Moysis, C. Volos, *Assessing the chaos strength of Taylor approximations of the sine chaotic map*, Nonlinear Dynamics 111 (2023) 2755–2778.
- L. Moysis et al., *A review of chaotification techniques for discrete-time systems*, Nonlinear Dynamics 113 (2025).
- S. Liao, *On the reliability of computed chaotic solutions of non-linear differential equations*, Tellus A 61 (2009).
- S. Liao, *Physical limit of prediction for chaotic motion of three-body problem*, CNSNS 19 (2014), arXiv:1305.6094.
- X. Li, S. Liao, *Clean numerical simulation…*, arXiv:1609.09344.
- P. Wang, J. Li, *On the relation between reliable computation time, float-point precision and the Lyapunov exponent in chaotic systems*, arXiv:1410.4919.
- B. Kehlet, A. Logg, *Quantifying the computability of the Lorenz system*, arXiv:1306.2782; and *A posteriori error analysis of round-off errors…*, Numer. Algorithms 76 (2017).
- T. Boekholt, S. Portegies Zwart, *On the reliability of N-body simulations*, ComAC 2 (2015), arXiv:1411.6671.
- S. Portegies Zwart, *Validating direct solvers for Newton's gravitational N-body problem, and the systematic comparison between IEEE floating point and Posits*, arXiv:2608.17032 (2026).
- M. Klöwer et al., *Periodic orbits in chaotic systems simulated at low precision*, Sci. Rep. 13 (2023).
- E. N. Lorenz, *Computational chaos*, Physica D 35 (1989); *Computational periodicity as observed in a simple system*, Tellus A 58 (2006).
- W. Hayes, K. Jackson, *A survey of shadowing methods for numerical solutions of ODEs*, Appl. Numer. Math. 53 (2005).
- D. Viswanath, *Global errors of numerical ODE solvers and Lyapunov's theory of stability*, IMA J. Numer. Anal. 21 (2001).
- À. Jorba, M. Zou, *A software package for the numerical integration of ODEs by means of high-order Taylor methods*, Exp. Math. 14 (2005).
- F. Johansson, *Arb*, IEEE Trans. Comput. 66 (2017), arXiv:1611.02831.
- N. J. Higham, T. Mary, *Mixed precision algorithms in numerical linear algebra*, Acta Numerica 31 (2022).
- E. Hairer, R. McLachlan, A. Razakarivony, *Achieving Brouwer's law with implicit Runge–Kutta methods*, BIT 48 (2008).
- A.-K. Kassam, L. N. Trefethen, *Fourth-order time-stepping for stiff PDEs*, SIAM J. Sci. Comput. 26 (2005).
- I. A. Baratta et al., *DOLFINx: the next generation FEniCS problem solving environment* (2023); M. S. Alnæs et al., *Unified Form Language*, ACM TOMS 40 (2014).
