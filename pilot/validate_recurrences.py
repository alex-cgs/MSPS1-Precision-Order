"""Validate the Taylor recurrences against SymPy Lie derivatives (exact rationals), orders 0..4."""
import sympy as sp, flint
from taylor3b import taylor_coeffs, initial_state, PAIRS
names = [f'qx{i}' for i in range(3)] + [f'qy{i}' for i in range(3)] + [f'vx{i}' for i in range(3)] + [f'vy{i}' for i in range(3)] + ['u01', 'u02', 'u12']
S = sp.symbols(names); qx, qy, vx, vy, u = S[0:3], S[3:6], S[6:9], S[9:12], S[12:15]
f = {}
for i in range(3):
    f[qx[i]] = vx[i]; f[qy[i]] = vy[i]
    ax = 0; ay = 0
    for p, (a, b) in enumerate(PAIRS):
        if i in (a, b):
            j = b if a == i else a
            ax += (qx[j]-qx[i])*u[p]**3; ay += (qy[j]-qy[i])*u[p]**3
    f[vx[i]] = ax; f[vy[i]] = ay
for p, (i, j) in enumerate(PAIRS):
    f[u[p]] = -u[p]**3*((qx[i]-qx[j])*(vx[i]-vx[j]) + (qy[i]-qy[j])*(vy[i]-vy[j]))
z0 = initial_state(); point = {S[k]: sp.Rational(z0[k].numerator, z0[k].denominator) for k in range(15)}
K = 4
c = taylor_coeffs([flint.fmpq(v.numerator, v.denominator) for v in z0], K)
ok = True
for idx, var in enumerate(S):
    g = var
    for k in range(K + 1):
        exact = sp.Rational(g.subs(point)) / sp.factorial(k)
        rec = sp.Rational(int(c[idx][k].p), int(c[idx][k].q))
        if exact != rec:
            ok = False; print('MISMATCH', var, k, exact, rec)
        g = sp.expand(sum(sp.diff(g, w)*f[w] for w in S))
print('All 15 variables x orders 0..%d match exactly:' % K, ok)
print('example: Taylor coefficients of u01 =', [str(c[12][k]) for k in range(K + 1)])
