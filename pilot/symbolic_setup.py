"""Step 1 – symbolic derivations for the planar three-body problem (SymPy)."""
import sympy as sp

m = sp.symbols('m1:4', positive=True)
x = sp.symbols('x1:4', real=True); y = sp.symbols('y1:4', real=True)
px = sp.symbols('px1:4', real=True); py = sp.symbols('py1:4', real=True)
pairs = [(0, 1), (0, 2), (1, 2)]
r = {pr: sp.sqrt((x[pr[0]]-x[pr[1]])**2 + (y[pr[0]]-y[pr[1]])**2) for pr in pairs}

# (a) Hamiltonian, G = 1
H = sum((px[i]**2 + py[i]**2)/(2*m[i]) for i in range(3)) - sum(m[i]*m[j]/r[(i, j)] for i, j in pairs)
eqs = {}
for i in range(3):
    eqs[f"x{i+1}'"] = sp.diff(H, px[i]); eqs[f"y{i+1}'"] = sp.diff(H, py[i])
    eqs[f"px{i+1}'"] = -sp.diff(H, x[i]); eqs[f"py{i+1}'"] = -sp.diff(H, y[i])
print("(a) Hamilton's equations, e.g.  px1' =", sp.simplify(eqs["px1'"]))

# (b) Polynomialization: u_ij = 1/r_ij, velocities v = p/m
t = sp.symbols('t')
X = [sp.Function(f'x{i+1}')(t) for i in range(3)]; Y = [sp.Function(f'y{i+1}')(t) for i in range(3)]
R = sp.sqrt((X[0]-X[1])**2 + (Y[0]-Y[1])**2)
du = sp.diff(1/R, t)
u, dx, dy, wx, wy = sp.symbols('u dx dy wx wy', real=True)
du_sub = du.subs({sp.diff(X[0], t)-sp.diff(X[1], t): wx}).subs({X[0]-X[1]: dx, Y[0]-Y[1]: dy})
du_sub = du_sub.subs({sp.Derivative(X[0], t): wx + sp.Derivative(X[1], t), sp.Derivative(Y[0], t): wy + sp.Derivative(Y[1], t)})
du_sub = sp.simplify(du_sub.subs({sp.sqrt(dx**2+dy**2): 1/u}))
print("(b) d/dt(1/r) with u = 1/r  ->", sp.factor(du_sub), "   [ = -u^3 (d.w) ]")
assert sp.simplify(du_sub + u**3*(dx*wx + dy*wy)) == 0

# (c) Lifted polynomial system and its invariants
qx = sp.symbols('qx1:4'); qy = sp.symbols('qy1:4'); vx = sp.symbols('vx1:4'); vy = sp.symbols('vy1:4')
U = {pr: sp.Symbol(f'u{pr[0]+1}{pr[1]+1}') for pr in pairs}
def uij(i, j): return U[(min(i, j), max(i, j))]
f = {}
for i in range(3):
    f[qx[i]] = vx[i]; f[qy[i]] = vy[i]
    f[vx[i]] = sum(m[j]*(qx[j]-qx[i])*uij(i, j)**3 for j in range(3) if j != i)
    f[vy[i]] = sum(m[j]*(qy[j]-qy[i])*uij(i, j)**3 for j in range(3) if j != i)
for (i, j) in pairs:
    s = (qx[i]-qx[j])*(vx[i]-vx[j]) + (qy[i]-qy[j])*(vy[i]-vy[j])
    f[U[(i, j)]] = -U[(i, j)]**3 * s
def Lie(F): return sp.expand(sum(sp.diff(F, k)*f[k] for k in f))
deg = max(sp.Poly(e, *f.keys()).total_degree() for e in f.values())
print(f"(c) lifted system: {len(f)} variables, polynomial RHS of total degree {deg}")
E = sum(m[i]*(vx[i]**2+vy[i]**2)/2 for i in range(3)) - sum(m[i]*m[j]*U[(i, j)] for i, j in pairs)
print("    dE/dt along lifted flow       =", sp.simplify(Lie(E)), " (exact invariant, no constraint needed)")
Px = sum(m[i]*vx[i] for i in range(3)); Lz = sum(m[i]*(qx[i]*vy[i]-qy[i]*vx[i]) for i in range(3))
print("    dPx/dt, dLz/dt                =", sp.simplify(Lie(Px)), ",", sp.simplify(Lie(Lz)))
i, j = 0, 1
d2 = (qx[i]-qx[j])**2 + (qy[i]-qy[j])**2; s12 = (qx[i]-qx[j])*(vx[i]-vx[j]) + (qy[i]-qy[j])*(vy[i]-vy[j])
C = U[(0, 1)]**2*d2 - 1
assert sp.simplify(Lie(C) + 2*U[(0, 1)]**2*s12*C) == 0
print("    constraint C = u^2 r^2 - 1:  dC/dt = -2 u^2 (d.w) C  -> C = 0 is invariant, errors in C are neither damped nor corrected")

# (d) Rational initial data: Pythagorean 3-4-5 triangle gives rational distances, so u_ij(0) is rational
q0 = [(1, 3), (-2, -1), (1, -1)]
for (a, b) in pairs:
    dist = sp.sqrt((q0[a][0]-q0[b][0])**2 + (q0[a][1]-q0[b][1])**2)
    print(f"(d) |q{a+1}-q{b+1}| = {dist}  ->  u{a+1}{b+1}(0) = {1/dist}")
