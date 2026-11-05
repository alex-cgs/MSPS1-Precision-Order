from style import BITS_LABEL, BITS_ORDER, MARKERS, bits_color, bits_marker, both, np, pd, plt, read, save, short_case


def _tpred_panels(t, tol, name, cases, theme, title):
    s, mod = read("e2_tpred_summary"), read("e2_model_tpred")
    if s is None:
        return
    s = s[(s.tol == tol) & s.case.isin(cases)]
    cases = [c for c in cases if c in set(s.case)]
    if not cases:
        return
    cols = 3
    rows = int(np.ceil(len(cases) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.1 * rows), squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    handles = {}
    for ax, c in zip(axs.flat, cases):
        ax.set_visible(True)
        d = s[s.case == c]
        for b in sorted(d.bits.unique()):
            g = d[d.bits == b].sort_values("n")
            col = bits_color(t, b)
            ax.fill_between(g.n, g.tpred_q25, g.tpred_q75, color=col, alpha=0.14, lw=0)
            h, = ax.plot(g.n, g.tpred_median, color=col, marker=bits_marker(b), ms=4, label=BITS_LABEL[b])
            handles[b] = h
            if mod is not None:
                m = mod[(mod.case == c) & (mod.bits == b) & (mod.tol == tol)].groupby("n").tpred_model.median()
                if len(m):
                    ax.plot(m.index, m.values, color=col, ls=":", lw=1)
        cens = d[d.n == d.n.max()].censored_frac.max()
        ax.set_title(f"{short_case(c)} (censored at top n: {cens:.0%})", fontsize=9.5)
        ax.set_xlabel("Taylor order n")
        ax.set_ylabel(f"T_pred, tol {tol:g}")
    labels = [handles[b] for b in sorted(handles)]
    labels.append(plt.Line2D([], [], color=t["INK2"], ls=":", lw=1, label="model (per-IC fit), median"))
    fig.legend(handles=labels, loc="lower center", ncol=min(8, len(labels)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=t["INK"], fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name, theme)


CASES = ["nb3_pilot", "nb3_ensemble:r4", "lorenz63:rho_28", "henon_heiles:E_1_8", "henon_heiles:E_1_6_r2", "rossler"]


@both
def tpred(theme, t):
    _tpred_panels(t, 1e-6, "e2_tpred_tol1e-6", CASES, theme, "Predictability time vs order per precision: median and IQR over initial conditions")
    _tpred_panels(t, 1e-3, "e2_tpred_tol1e-3", CASES, theme, "Same, tolerance 1e-3")


@both
def crossover(theme, t):
    cs = read("e2_crossover_summary")
    if cs is None:
        return
    cs = cs[cs.tol == 1e-6].dropna(subset=["n_star_model_median"])
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    cases = [c for c in CASES if c in set(cs.case)]
    fam = {c: i for i, c in enumerate(cases)}
    offs = np.linspace(-3, 3, len(cases))
    for c in cases:
        i = fam[c]
        g = cs[cs.case == c].sort_values("bits")
        col = t["CAT"][i % 4]
        style = "-" if i < 4 else "--"
        ax.errorbar(g.bits + offs[i], g.n_star_model_median, yerr=[g.n_star_model_median - g.model_q25, g.model_q75 - g.n_star_model_median], color=col, ls=style, marker=MARKERS[i], ms=5, capsize=2, lw=1.4, label=short_case(c))
    ax.axhspan(4, 32, color=t["BAND"], alpha=0.5, lw=0, zorder=0)
    ax.text(ax.get_xlim()[1], 32.6, "swept orders n ≤ 32", ha="right", fontsize=8, color=t["INK2"])
    ax.set_xlabel("precision p (bits)")
    ax.set_ylabel("crossover order n*(p), model-based")
    ax.set_title("Crossover order grows linearly with precision across systems")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "e2_crossover", theme)


@both
def pred_vs_meas(theme, t):
    p = read("e3_points_sample")
    if p is None:
        return
    cases = sorted(p.case.unique())
    cols = 3
    rows = int(np.ceil(len(cases) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    tab = read("e3_model_out_of_sample").set_index("case")
    for ax, c in zip(axs.flat, cases):
        ax.set_visible(True)
        d = p[p.case == c]
        for b in sorted(d.bits.unique()):
            g = d[d.bits == b]
            ax.scatter(g.log10E_pred_minus_logA, g.log10E_measured_minus_logA, s=5, alpha=0.5, color=bits_color(t, b), label=BITS_LABEL[b], lw=0)
        lo = min(d.log10E_pred_minus_logA.min(), d.log10E_measured_minus_logA.min())
        hi = max(d.log10E_pred_minus_logA.max(), d.log10E_measured_minus_logA.max())
        ax.plot([lo, hi], [lo, hi], color=t["MUTED"], lw=1, ls="--")
        ax.plot([lo, hi], [lo + 1, hi + 1], color=t["MUTED"], lw=0.6, ls=":")
        ax.plot([lo, hi], [lo - 1, hi - 1], color=t["MUTED"], lw=0.6, ls=":")
        r = tab.loc[c]
        ax.set_title(f"{short_case(c)}: rms {r.test_rms:.2f}, FTLE model {r.ftle_test_rms:.2f}", fontsize=9)
        ax.set_xlabel("predicted log10 E/A")
        ax.set_ylabel("measured log10 E/A")
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.02), markerscale=2.5)
    fig.suptitle("Out-of-sample model check: fit on half the initial conditions, tested on the other half (dotted: ±1 decade)", color=t["INK"], fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "e3_pred_vs_measured", theme)


if __name__ == "__main__":
    tpred(); crossover(); pred_vs_meas()
