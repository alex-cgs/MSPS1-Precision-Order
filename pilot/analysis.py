"""Pilot analysis: error-model fit and figures (light and dark variants). Reads results.json and ref.pkl."""
import json, math, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

R = json.load(open('results.json'))
OUT = 'fig'; os.makedirs(OUT, exist_ok=True)
TOL = 1e-6
BITS_LIST = (53, 80, 113, 160)
LABEL = {53: 'float64 (53 bits, ≈16 digits)', 80: '80 bits (≈24 digits)', 113: '113 bits (quad, ≈34 digits)', 160: '160 bits (≈48 digits)'}
THEMES = {
 'light': dict(INK='#0b0b0b', INK2='#52514e', MUTED='#898781', GRID='#e1e0d9', BASE='#c3c2b7', SURF='#fcfcfb',
               CAT=['#2a78d6', '#eb6834', '#1baf7a'], BAND='#cde2fb',
               SEQ={53: '#86b6ef', 80: '#3987e5', 113: '#1c5cab', 160: '#0d366b'},
               RAMP=['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']),
 'dark': dict(INK='#f0efec', INK2='#c3c2b7', MUTED='#898781', GRID='#2c2c2a', BASE='#383835', SURF='#1a1a19',
              CAT=['#3987e5', '#d95926', '#199e70'], BAND='#184f95',
              SEQ={53: '#184f95', 80: '#256abf', 113: '#5598e7', 160: '#9ec5f4'},
              RAMP=['#184f95', '#256abf', '#3987e5', '#6da7ec', '#b7d3f6']),
}

summary = {}
sched, ref = pickle.load(open('ref.pkl', 'rb'))
T = float(sum(float(h) for h in sched))
Z = np.array([[float(v) for v in z[:6]] for z in ref])
qx, qy = Z[:, :3], Z[:, 3:6]
dists = [np.hypot(qx[:, i] - qx[:, j], qy[:, i] - qy[:, j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
summary['dmin'] = float(min(d.min() for d in dists)); summary['dmax'] = float(max(d.max() for d in dists))
h = np.array([float(x) for x in sched]); summary.update(steps=len(h), hmin=h.min(), hmax=h.max(), T=T)
summary['ref'] = R['ref']; summary['ref_check'] = R['ref_check']

# amplification A(t) of a unit perturbation (velocity perturbed by PERT, so position separation starts at 0)
ly_t = np.array(R['lyap']['t']); ly_s = np.array(R['lyap']['sep'])
PERT = 1e-20
A = np.maximum(ly_s, PERT * 1e-3) / PERT
mask = ly_t > 5
lam = np.polyfit(ly_t[mask], np.log(A[mask]), 1)[0]
summary.update(lambda_fit=lam, amplification_T=float(A[-1]), log10_amp_T=float(np.log10(A[-1])))

# local truncation law
ns = sorted(int(k) for k in R['local'])
med = np.array([np.median(R['local'][str(n)]) for n in ns])
q1 = np.array([np.percentile(R['local'][str(n)], 25) for n in ns]); q3 = np.array([np.percentile(R['local'][str(n)], 75) for n in ns])
fm = med > -110
slope, icpt = np.polyfit(np.array(ns)[fm] + 1, med[fm], 1)
summary.update(local_slope_per_order=slope, local_icpt=icpt, rho_over_h_effective=10 ** (-slope),
               local_median=dict(zip(ns, med.round(2).tolist())))

runs = R['sweep']
def tpred(s):
    e = np.array(s['err']); t = np.array(s['t'])
    idx = np.where(e > TOL)[0]
    if len(idx): return float(t[idx[0]]), False
    if s['t'][-1] < T - 1e-9: return float(s['t'][-1]), False
    return T, True
rows = []
for s in runs:
    tp, cens = tpred(s)
    ok = s['t'][-1] >= T - 1e-9
    rows.append(dict(n=s['n'], bits=s['bits'], tpred=tp, censored=cens, errT=s['err'][-1] if ok else float('nan'),
                     wall=s['wall'], dE=max(s['dE']), dC=max(s['dC']), reached=s['t'][-1], blew_up=not ok))
summary['rows'] = rows

def logA_at(t): return np.interp(t, ly_t, np.log10(A))

# two-constant error model:  E(t) ≈ [10^ct · eps_trunc(n) + 10^cr · 2^-bits] · A(t)
obs = []
for s_ in runs:
    t = np.array(s_['t']); e = np.array(s_['err'])
    sel = (t > 10) & (e > 0) & (e < 1e-2) & np.isfinite(e)
    if sel.sum() < 5: continue
    for ti, ei in zip(t[sel][::10], e[sel][::10]):
        obs.append((s_['n'], s_['bits'], np.log10(ei) - logA_at(ti)))
obs = np.array(obs)
def model(cc, n, b): return np.log10(10 ** (cc[0] + icpt + slope * (n + 1)) + 10 ** (cc[1] - b * math.log10(2)))
def loss(cc): return np.mean((obs[:, 2] - model(cc, obs[:, 0], obs[:, 1])) ** 2)
fit = minimize(loss, [7.0, 7.0], method='Nelder-Mead')
ct, cr = fit.x
summary.update(c_trunc=float(ct), c_round=float(cr), fit_rms=float(np.sqrt(fit.fun)), fit_obs=len(obs))
summary['crossover'] = {b: (cr - b * math.log10(2) - ct - icpt) / slope - 1 for b in BITS_LIST}
def pred_logE(n, bits, t): return float(model((ct, cr), n, bits)) + logA_at(t)
def pred_tpred(n, bits):
    v = np.array([pred_logE(n, bits, t) for t in ly_t]); idx = np.where(v > math.log10(TOL))[0]
    return float(ly_t[idx[0]]) if len(idx) else T
NL = sorted(set(r['n'] for r in rows))
summary['pred_tpred'] = {f'{n}_{b}': pred_tpred(n, b) for n in NL for b in BITS_LIST}
pm = [(pred_logE(r['n'], r['bits'], T), math.log10(r['errT']), r['bits']) for r in rows
      if r['errT'] == r['errT'] and 0 < r['errT'] < 1]
res = np.array([m - p for p, m, _ in pm])
summary.update(pred_resid_rms=float(np.sqrt(np.mean(res ** 2))), pred_resid_max=float(np.max(np.abs(res))), pred_points=len(pm))
ex = R['exact']
summary['exact'] = dict(n=ex['n'], bits=ex['bits'], wall=ex['wall'], steps=ex['steps'], t_reached=ex['t_reached'], err=ex['err_vs_ref'],
                        growth=[ex['bits'][i + 1] / ex['bits'][i] for i in range(len(ex['bits']) - 1)])


def draw(theme):
    th = THEMES[theme]; INK, INK2, MUTED, SURF = th['INK'], th['INK2'], th['MUTED'], th['SURF']
    CAT, SEQ, RAMP = th['CAT'], th['SEQ'], th['RAMP']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': th['BASE'], 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': th['GRID'], 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                         'savefig.facecolor': SURF, 'lines.linewidth': 2, 'legend.frameon': False, 'axes.titlecolor': INK,
                         'axes.titlesize': 11, 'text.color': INK2, 'legend.labelcolor': INK2})
    sfx = f'_{theme}.png'

    fig, ax = plt.subplots(figsize=(6, 5.2))
    for i in range(3):
        ax.plot(Z[:, i], Z[:, 3 + i], color=CAT[i], lw=0.8, label=f'body {i+1}')
        ax.plot(Z[0, i], Z[0, 3 + i], 'o', color=CAT[i], ms=9, mec=SURF, mew=2)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Reference orbit for t in [0, {T:.0f}]; dots mark the start')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig1_orbit' + sfx, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(ns, q1, q3, color=th['BAND'], lw=0, label='middle 50% of 60 sampled steps')
    ax.plot(ns, med, color=CAT[0], marker='o', ms=5, label='median one-step error')
    xx = np.linspace(ns[0], ns[-1], 50)
    ax.plot(xx, icpt + slope * (xx + 1), color=INK2, ls='--', lw=1.2, label=f'fit: log10 ε = {icpt:.1f} {slope:+.2f}(n+1)')
    for b in (53, 113, 160):
        y = -b * math.log10(2)
        ax.axhline(y, color=MUTED, lw=0.8, ls=':')
        xpos, ha = (ns[-1], 'right') if b == 53 else (ns[0], 'left')
        ax.text(xpos, y - 0.6, f'2^-{b}  ({b}-bit rounding unit)', ha=ha, va='top', color=INK2, fontsize=8.5)
    ax.set_xlabel('Taylor order n'); ax.set_ylabel('log10 one-step position error')
    ax.set_title('Truncation error per step falls by a fixed factor per order')
    ax.legend(loc='upper right', fontsize=8.5)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig2_local_error' + sfx, dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, bits in zip(axs, (53, 160)):
        for n, col in zip([6, 10, 14, 20, 28], RAMP):
            s = next(x for x in runs if x['n'] == n and x['bits'] == bits)
            ax.semilogy(s['t'], np.maximum(np.array(s['err']), 1e-60), color=col, lw=1.3, label=f'n = {n}')
        ax.semilogy(ly_t, A * 1e-40, color=MUTED, ls='--', lw=1.2, label='growth of a tiny perturbation (shape only)')
        ax.axhline(TOL, color=INK2, lw=0.8, ls=':'); ax.text(1, TOL * 3, f'tolerance {TOL:g}', color=INK2, fontsize=8.5)
        ax.set_title(LABEL[bits]); ax.set_xlabel('time t'); ax.set_ylim(1e-50, 1e3)
    axs[0].set_ylabel('max position error vs reference')
    hh, ll = axs[1].get_legend_handles_labels()
    fig.legend(hh, ll, loc='lower center', ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f'{OUT}/fig3_error_growth' + sfx, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for bits in BITS_LIST:
        rr = sorted([r for r in rows if r['bits'] == bits], key=lambda r: r['n'])
        ax.plot([r['n'] for r in rr], [r['tpred'] for r in rr], color=SEQ[bits], marker='o', ms=5, label=LABEL[bits])
        for r in rr:
            if r['censored']:
                ax.plot(r['n'], r['tpred'], marker='^', color=SEQ[bits], ms=9, mec=SURF, mew=1.5)
        ax.plot(NL, [pred_tpred(n, bits) for n in NL], color=SEQ[bits], ls='--', lw=1)
    ax.set_xlabel('Taylor order n'); ax.set_ylabel(f'time until error > {TOL:g}')
    ax.set_title('Predictability time: measured (solid) vs model (dashed)')
    ax.text(0.01, 0.98, '▲ never crossed the tolerance before T', transform=ax.transAxes, va='top', fontsize=8.5, color=INK2)
    ax.set_xticks(NL); ax.legend(loc='lower right', fontsize=8.5)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig4_tpred' + sfx, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5))
    for bits in BITS_LIST:
        pts = [(p, m) for p, m, b in pm if b == bits]
        if pts:
            p, m = zip(*pts); ax.plot(p, m, 'o', color=SEQ[bits], ms=7, mec=SURF, mew=1.5, label=LABEL[bits])
    allv = [p for p, _, _ in pm] + [m for _, m, _ in pm]
    lo, hi = min(allv) - 1, max(allv) + 1
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls='--', lw=1)
    ax.set_xlabel('predicted log10 error at t = T'); ax.set_ylabel('measured log10 error at t = T')
    ax.set_title('Model check across all runs'); ax.legend(fontsize=8.5, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig5_pred_vs_meas' + sfx, dpi=150); plt.close(fig)

    k = np.arange(1, len(ex['bits']) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.semilogy(k, ex['bits'], color=CAT[0], marker='o', ms=7, label='measured: largest numerator + denominator')
    ax.semilogy(k, [ex['bits'][0] * (4 * ex['n'] + 1) ** i for i in range(len(k))], color=INK2, ls='--', lw=1.2,
                label=f'prediction: ×(4n+1) = ×{4*ex["n"]+1} per step')
    ax.axhline(53, color=MUTED, ls=':', lw=0.8); ax.text(k[-1], 70, 'float64: 53 bits', ha='right', color=INK2, fontsize=8.5)
    ax.set_xticks(k); ax.set_xlabel('step'); ax.set_ylabel('bits to store the state exactly')
    ax.set_title(f'Exact rational arithmetic, n = {ex["n"]}: storage per step')
    ax.legend(fontsize=8.5, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig6_exact_bits' + sfx, dpi=150); plt.close(fig)


for th in THEMES: draw(th)
json.dump(summary, open('summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'pred_tpred')}, indent=1, default=float))
