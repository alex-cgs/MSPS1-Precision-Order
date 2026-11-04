"""Taylor method of order n for the polynomialized planar 3-body problem (equal masses, G = 1).

One code path; the number type is swapped to change the arithmetic:
  'float'    : IEEE double (53-bit significand)
  'mpfr:<b>' : gmpy2 MPFR with b-bit significand
  'exact'    : python-flint fmpq exact rationals

State (15 variables): qx[3], qy[3], vx[3], vy[3], u[3] with u_ij = 1/|q_i - q_j|, pairs (0,1), (0,2), (1,2).
ODE:  q_i' = v_i,  v_i' = sum_j (q_j - q_i) u_ij^3,  u_ij' = -u_ij^3 (q_i - q_j).(v_i - v_j)
"""
from fractions import Fraction
import math, time

PAIRS = [(0, 1), (0, 2), (1, 2)]

# ------------------------------------------------------------------ initial data (all rational)
Q0 = [(1, 3), (-2, -1), (1, -1)]                       # 3-4-5 triangle: distances 5, 4, 3
V0 = [(Fraction(0), Fraction(-1, 4)), (Fraction(1, 4), Fraction(1, 4)), (Fraction(-1, 4), Fraction(0))]
U0 = [Fraction(1, 5), Fraction(1, 4), Fraction(1, 3)]


def initial_state():
    """Exact initial state as Fractions in the order qx, qy, vx, vy, u."""
    qx = [Fraction(q[0]) for q in Q0]; qy = [Fraction(q[1]) for q in Q0]
    vx = [v[0] for v in V0]; vy = [v[1] for v in V0]
    return qx + qy + vx + vy + list(U0)


# ------------------------------------------------------------------ arithmetic back-ends
def make_converter(arith):
    if arith == 'float':
        return lambda fr: float(fr)
    if arith.startswith('mpfr:'):
        import gmpy2
        gmpy2.get_context().precision = int(arith.split(':')[1])
        return lambda fr: gmpy2.mpfr(gmpy2.mpq(fr.numerator, fr.denominator))
    if arith == 'exact':
        import flint
        return lambda fr: flint.fmpq(fr.numerator, fr.denominator)
    raise ValueError(arith)


# ------------------------------------------------------------------ Taylor coefficients
def taylor_coeffs(z, n):
    """Return list of 15 coefficient lists c[var][0..n] for the solution through state z."""
    qx = [[z[i]] for i in range(3)]; qy = [[z[3 + i]] for i in range(3)]
    vx = [[z[6 + i]] for i in range(3)]; vy = [[z[9 + i]] for i in range(3)]
    u = [[z[12 + p]] for p in range(3)]
    dx = [[] for _ in PAIRS]; dy = [[] for _ in PAIRS]; wx = [[] for _ in PAIRS]; wy = [[] for _ in PAIRS]
    u2 = [[] for _ in PAIRS]; u3 = [[] for _ in PAIRS]; s = [[] for _ in PAIRS]
    us = [[] for _ in PAIRS]; ax = [[] for _ in PAIRS]; ay = [[] for _ in PAIRS]
    for k in range(n):
        for p, (i, j) in enumerate(PAIRS):
            dx[p].append(qx[i][k] - qx[j][k]); dy[p].append(qy[i][k] - qy[j][k])
            wx[p].append(vx[i][k] - vx[j][k]); wy[p].append(vy[i][k] - vy[j][k])
            U, U2, DX, DY, WX, WY = u[p], u2[p], dx[p], dy[p], wx[p], wy[p]
            r = range(k + 1)
            u2[p].append(sum(U[l] * U[k - l] for l in r))
            U3 = u3[p]; U3.append(sum(U2[l] * U[k - l] for l in r))
            S = s[p]; S.append(sum(DX[l] * WX[k - l] + DY[l] * WY[k - l] for l in r))
            us[p].append(sum(U3[l] * S[k - l] for l in r))
            ax[p].append(sum(DX[l] * U3[k - l] for l in r))
            ay[p].append(sum(DY[l] * U3[k - l] for l in r))
        k1 = k + 1
        for i in range(3):
            qx[i].append(vx[i][k] / k1); qy[i].append(vy[i][k] / k1)
            fx = 0; fy = 0
            for p, (a, b) in enumerate(PAIRS):
                if a == i:
                    fx -= ax[p][k]; fy -= ay[p][k]
                elif b == i:
                    fx += ax[p][k]; fy += ay[p][k]
            vx[i].append(fx / k1); vy[i].append(fy / k1)
        for p in range(3):
            u[p].append(-us[p][k] / k1)
    return qx + qy + vx + vy + u


def horner(c, h):
    acc = c[-1]
    for a in reversed(c[:-1]):
        acc = acc * h + a
    return acc


def radius_estimate(c):
    """Radius of convergence estimate from the last two coefficients (Jorba–Zou style)."""
    n = len(c[0]) - 1
    est = []
    for k in (n - 1, n):
        mk = max(abs(float(v[k])) for v in c)
        if mk > 0:
            est.append(mk ** (-1.0 / k))
    return min(est) if est else float('inf')


# ------------------------------------------------------------------ diagnostics
def energy(z):
    vx, vy, u = z[6:9], z[9:12], z[12:15]
    return sum(vx[i] * vx[i] + vy[i] * vy[i] for i in range(3)) / 2 - (u[0] + u[1] + u[2])


def constraint_drift(z):
    qx, qy, u = z[0:3], z[3:6], z[12:15]
    out = 0.0
    for p, (i, j) in enumerate(PAIRS):
        try:
            dx, dy = qx[i] - qx[j], qy[i] - qy[j]
            out = max(out, abs(float(u[p] * u[p] * (dx * dx + dy * dy) - 1)))
        except (OverflowError, ValueError):
            return float('inf')
    return out


# ----------------------------------------------------------------- integrators
def make_schedule(T, n=40, bits=300, safety=8):
    """Reference run that also builds the shared dyadic step schedule h_k = 2^floor(log2(rho/safety))."""
    conv = make_converter(f'mpfr:{bits}')
    z = [conv(v) for v in initial_state()]
    t = Fraction(0); sched = []; states = [z]
    while t < T:
        c = taylor_coeffs(z, n)
        rho = radius_estimate(c)
        h = Fraction(2) ** math.floor(math.log2(rho / safety))
        h = min(h, Fraction(1, 2), Fraction(T) - t)
        hc = conv(h)
        z = [horner(ci, hc) for ci in c]
        t += h; sched.append(h); states.append(z)
    return sched, states


def integrate(n, arith, sched, max_steps=None, record=True, time_budget=None, z0=None):
    conv = make_converter(arith)
    z = [conv(v) for v in (z0 if z0 is not None else initial_state())]
    hs = [conv(h) for h in sched]
    states = [z] if record else None
    t0 = time.perf_counter()
    for k, h in enumerate(hs):
        if max_steps is not None and k >= max_steps:
            break
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            break
        c = taylor_coeffs(z, n)
        z = [horner(ci, h) for ci in c]
        if record:
            states.append(z)
        if any(not math.isfinite(float(v)) or abs(float(v)) > 1e8 for v in z[:12]):
            break  # blow-up
    return states, time.perf_counter() - t0
