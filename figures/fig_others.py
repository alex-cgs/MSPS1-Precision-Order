from style import BITS_LABEL, DASHES, MARKERS, ROOT, bits_color, bits_marker, both, np, pd, plt, read, save, short_case


@both
def stats_vs_traj(theme, t):
    d = read("e5_stats_summary")
    if d is None:
        return
    cases = sorted(d.case.unique())
    fig, axs = plt.subplots(len(cases), 3, figsize=(11.5, 2.6 * len(cases)), squeeze=False)
    for row, c in zip(axs, cases):
        g = d[(d.case == c) & (~d.is_reference)]
        ob = next(k for k in d.columns if k.startswith("W1_") and "floor" not in k and d[d.case == c][k].notna().any())
        for b in sorted(g.bits.unique()):
            h = g[g.bits == b].sort_values("n")
            col = bits_color(t, b)
            row[0].plot(h.n, h["tpred_median_1e-6"], color=col, marker=bits_marker(b), ms=4, label=BITS_LABEL[b])
            row[1].plot(h.n, np.maximum(h[ob], 1e-6), color=col, marker=bits_marker(b), ms=4)
            row[2].plot(h.n, np.maximum(h.lambda_abs_err, 1e-5), color=col, marker=bits_marker(b), ms=4)
        floor = d[d.case == c]["W1_floor_" + ob[3:]].iloc[0]
        if floor == floor:
            row[1].axhline(max(floor, 1e-6), color=t["MUTED"], ls="--", lw=1)
            row[1].text(0.99, 0.97, "reference split-half floor", transform=row[1].transAxes, ha="right", va="top", fontsize=7.5)
        row[0].set_title(f"{short_case(c)}: time on orbit", fontsize=9.5)
        row[0].set_ylabel("T_pred (tol 1e-6)")
        row[1].set_yscale("log"); row[2].set_yscale("log")
        row[1].set_title(f"histogram W1, {ob[3:]}", fontsize=9.5)
        row[2].set_title("|λ − λ_ref|", fontsize=9.5)
        for ax in row:
            ax.set_xlabel("order n")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Statistics against time on the orbit, same runs: per-run adaptive step, reference = highest (n, p) in each grid", color=t["INK"], fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    save(fig, "e5_stats_vs_trajectories", theme)


@both
def escapes(theme, t):
    d = read("e5_stats_summary")
    if d is None or "esc_ks" not in d:
        return
    g = d[d.case == "nb3_ensemble"]
    if not len(g):
        return
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for b in sorted(g.bits.unique()):
        h = g[g.bits == b].sort_values("n")
        col = bits_color(t, b)
        axs[0].plot(h.n, h.esc_frac, color=col, marker=bits_marker(b), ms=4, label=BITS_LABEL[b])
        axs[1].plot(h.n, h.esc_ks, color=col, marker=bits_marker(b), ms=4)
        axs[2].plot(h.n, h.esc_body_TV, color=col, marker=bits_marker(b), ms=4)
    axs[0].set_ylabel("fraction of runs that escape"); axs[1].set_ylabel("KS distance, escape times"); axs[2].set_ylabel("TV distance, escaping body")
    for ax in axs:
        ax.set_xlabel("order n")
    axs[0].legend()
    fig.suptitle("Three-body escape statistics against the reference ensemble", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e5_escapes", theme)


@both
def representation(theme, t):
    o, m = read("e9_odes"), read("e9_maps")
    if o is None and m is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
    kinds = {"f64": (0, "o", "-"), "mpfr-53": (1, "s", "--"), "qr-53": (2, "^", ":"), "f32": (0, "o", "-"), "mpfr-24": (1, "s", "--"), "qr-24": (2, "^", ":")}
    names = {"f64": "IEEE float", "mpfr-53": "MPFR, same bits", "qr-53": "rational rounded to 2^-p", "f32": "", "mpfr-24": "", "qr-24": ""}
    for ax, c in zip(axs[:2], ["lorenz63:rho_28", "nb3_pilot"]):
        if o is None:
            continue
        d = o[(o.case == c) & (o.tol == 1e-6)]
        for a in ("f64", "mpfr-53", "qr-53", "f32", "mpfr-24", "qr-24"):
            g = d[d.arith == a].sort_values("n")
            if not len(g):
                continue
            i, mk, ls = kinds[a]
            ax.plot(g.n, g.tpred_median, color=t["CAT"][i], marker=mk, ls=ls, ms=4 if g.bits.iloc[0] == 53 else 3, alpha=1 if g.bits.iloc[0] == 53 else 0.55, label=names[a] or None)
        ax.set_title(short_case(c) + " (faint: 24 bits, solid: 53 bits)", fontsize=9.5)
        ax.set_xlabel("order n"); ax.set_ylabel("T_pred (tol 1e-6)")
    axs[0].legend(fontsize=8)
    ax = axs[2]
    if m is not None:
        d = m[m.bits == 53]
        cases = sorted(d.case.unique())
        x = np.arange(len(cases))
        for j, (a, lab) in enumerate((("f64", "IEEE float"), ("mpfr-53", "MPFR"), ("qr-53", "rational"))):
            v = [d[(d.case == c) & (d.arith == a)].k_cross_median.max() for c in cases]
            ax.bar(x + (j - 1) * 0.26, np.nan_to_num(v), width=0.24, color=t["CAT"][j], label=lab)
        ax.set_xticks(x, [c.replace("_map", "").replace(":mu_", " μ=").replace("199_100", "1.99") for c in cases], fontsize=8)
        ax.set_ylabel("iterations on orbit (tol 1e-6), p = 53")
        ax.set_title("Maps", fontsize=9.5)
    fig.suptitle("Representation vs bit count", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e9_representation", theme)


@both
def methods(theme, t):
    d = read("e8_methods")
    if d is None:
        return
    d = d[d.arith.isin(["f64"])]
    cases = [c for c in ["nb3_pilot", "henon_heiles:E_1_8", "fput_beta:N_16", "lorenz63:rho_28"] if c in set(d.case)]
    fam = {"taylor": (1, "o", "Taylor"), "gl": (2, "^", "Gauss–Legendre"), "yoshida": (3, "D", "Yoshida/Verlet"), "rk4": (0, "s", "RK4"), "dop853": (0, "v", "DOP853")}
    fig, axs = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 5.6), squeeze=False)
    for j, c in enumerate(cases):
        g = d[d.case == c]
        dH = [k for k in g.columns if k.startswith("max_d") and k not in ("max_dC",) and g[k].notna().any()]
        for m, (ci, mk, lab) in fam.items():
            s = g[g.method == m]
            if not len(s):
                continue
            axs[0, j].scatter(s.wall, s["tpred_1e-6"], color=t["CAT"][ci], marker=mk, s=22, label=lab, edgecolor=t["SURF"], lw=0.4)
            if "max_denergy" in g and s["max_denergy"].notna().any():
                axs[1, j].scatter(s.wall, np.maximum(s["max_denergy"], 1e-18), color=t["CAT"][ci], marker=mk, s=22, edgecolor=t["SURF"], lw=0.4)
        axs[0, j].set_xscale("log"); axs[1, j].set_xscale("log"); axs[1, j].set_yscale("log")
        axs[0, j].set_title(short_case(c), fontsize=9.5)
        axs[1, j].set_xlabel("wall time (s)")
    axs[0, 0].set_ylabel("T_pred (tol 1e-6)")
    axs[1, 0].set_ylabel("max |ΔH|")
    axs[0, 0].legend(fontsize=7.5)
    fig.suptitle("Method families, f64: accuracy and energy against cost", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e8_cost_vs_accuracy", theme)


@both
def hardware(theme, t):
    d, a = read("e10_throughput"), read("e10_accuracy")
    if d is None:
        return
    d = d.dropna(subset=["throughput_median"])
    d = d[d["size"] != "tiny"] if (d["size"] != "tiny").any() else d
    labels = sorted(d.label.unique())
    cases = sorted(d.case.unique())
    nsel = 16 if (d.n == 16).any() else d.n.max()
    fig, axs = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 3.9), squeeze=False)
    for ax, c in zip(axs[0], cases):
        for i, l in enumerate(labels):
            for j, p in enumerate(("f32", "f64", "dd")):
                g = d[(d.case == c) & (d.label == l) & (d.prec == p) & (d.n == nsel)].sort_values("batch")
                if len(g):
                    ax.plot(g.batch, g.throughput_median, color=t["CAT"][i % 4], ls=DASHES[j], marker=MARKERS[j], ms=4, label=f"{l}, {p}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("ensemble size"); ax.set_ylabel("trajectory-steps per second")
        ax.set_title(f"{short_case(c)}, n = {nsel}", fontsize=9.5)
        ax.legend(fontsize=7)
    fig.suptitle("Hardware throughput (warm-up excluded, median of repeats)", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e10_throughput", theme)
    if a is not None and "cost_per_accurate_time_s" in a:
        a = a.dropna(subset=["cost_per_accurate_time_s"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        labs = sorted(a.label.unique())
        mk = {"f32": "o", "f64": "s", "dd": "^"}
        for i, l in enumerate(labs):
            for p in mk:
                g = a[(a.label == l) & (a.prec == p)]
                ax.scatter(g["tpred_median_1e-06"], g.cost_per_accurate_time_s, color=t["CAT"][i % 4], marker=mk[p], s=30, label=f"{l}, {p}", edgecolor=t["SURF"], lw=0.4)
        ax.set_yscale("log"); ax.set_xlabel("median T_pred (tol 1e-6)"); ax.set_ylabel("wall s per member per accurate time unit")
        ax.legend(fontsize=7); ax.set_title("Cost per accurate time unit, one member at a time on small ensembles", fontsize=9.5)
        fig.tight_layout()
        save(fig, "e10_cost_accuracy", theme)


@both
def spurious(theme, t):
    per, dia = read("e6_map_periodicity"), read("e6_map_diagrams")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axs[0, 0]
    if per is not None:
        order = ["f16", "mpfr-16", "f32", "mpfr-24", "mpfr-32"]
        for i, c in enumerate(sorted(per.case.unique())):
            g = per[per.case == c].set_index("arith").reindex(order).dropna(subset=["period_median"])
            ax.errorbar(range(len(g)), g.period_median, yerr=[g.period_median - g.period_q25, g.period_q75 - g.period_median], color=t["CAT"][i], marker=MARKERS[i], capsize=2, label=c.replace("_map", ""))
        ax.set_xticks(range(len(order)), order, fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("orbit period reached (iterations)")
        ax.set_title("Low precision turns chaotic maps periodic", fontsize=9.5); ax.legend(fontsize=8)
    ax = axs[0, 1]
    from msps.io import iter_runs, load_data
    drawn = False
    w_styles = {None: ("true sin", t["INK2"], "-"), 1: ("T_1", t["CAT"][1], ":"), 2: ("T_2", t["CAT"][0], "--"), 4: ("T_4", t["CAT"][2], "-.")}
    for d_, m in iter_runs("E6"):
        c = m.get("config", {})
        if c.get("kind") == "map_diagram" and c.get("builder") == "sine" and c.get("arith") == "f64" and c.get("w") in w_styles:
            df = load_data(d_)
            lab, col, ls = w_styles[c.get("w")]
            ax.plot(df.mu, np.where(df.bounded, df["lambda"], np.nan), color=col, ls=ls, lw=1, label=lab)
            drawn = True
    if drawn:
        ax.axhline(0, color=t["MUTED"], lw=0.8)
        ax.set_ylim(-3, 1.2); ax.set_xlabel("μ"); ax.set_ylabel("Lyapunov exponent")
        ax.set_title("Sine map and Taylor truncations (f64)", fontsize=9.5); ax.legend(fontsize=8)
    ax = axs[1, 0]
    duf = read("e6_duffing_regimes")
    if duf is not None:
        dd = duf[duf.arith == "f64"]
        gammas = sorted(dd.case.unique())
        ns = sorted(dd.n.unique())
        M = np.array([[dd[(dd.case == g) & (dd.n == n)].distinct_median.max() for n in ns] for g in gammas], float)
        im = ax.imshow(np.log10(np.maximum(M, 1)), aspect="auto", cmap="Blues_r" if theme == "dark" else "Blues", origin="lower")
        ax.set_xticks(range(len(ns)), ns); ax.set_yticks(range(len(gammas)), [g.split("gamma_")[1].replace("_", ".") for g in gammas])
        ax.set_xlabel("order n (fixed step P/64)"); ax.set_ylabel("γ")
        for i in range(len(gammas)):
            for j in range(len(ns)):
                ax.text(j, i, f"{int(M[i, j])}" if M[i, j] == M[i, j] else "–", ha="center", va="center", fontsize=7.5, color=t["INK"] if M[i, j] < 20 else t["SURF"])
        ax.grid(False); ax.set_title("Duffing: distinct stroboscopic values, f64 (true periods need n ≥ ...)", fontsize=9.5)
    ax = axs[1, 1]
    drawn = False
    secs = {}
    for d_, m in iter_runs("E6"):
        c = m.get("config", {})
        if c.get("kind") == "poincare" and c.get("case") == "henon_heiles:E_1_8":
            key = (c["n"], c["arith"])
            if key in ((24, "dd"), (3, "f32")):
                secs.setdefault(key, []).append(load_data(d_))
    for (key, frames), col, lab, sz in zip(sorted(secs.items(), reverse=True), (t["CAT"][0], t["CAT"][1]), ("n=24, dd", "n=3, f32"), (0.4, 0.6)):
        df = pd.concat(frames)
        ax.scatter(df.y, df.py, s=sz, color=col, lw=0, label=lab, alpha=0.7)
        drawn = True
    if drawn:
        ax.set_xlabel("y"); ax.set_ylabel("p_y"); ax.set_title("Hénon–Heiles section E = 1/8", fontsize=9.5)
        ax.set_xlim(-0.6, 1.0); ax.set_ylim(-0.6, 0.6)
        ax.legend(markerscale=8, fontsize=8)
    fig.suptitle("Spurious dynamics at low precision and low order", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e6_spurious", theme)


@both
def dimension(theme, t):
    d = read("e11_dimension")
    if d is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    fam = {"nbN_circle": (0, "o", "circle"), "nbN_alternating_rings": (1, "s", "two rings"), "nbN_alternating_hier": (1, "v", "binary + perturbers"), "nbN_cluster": (2, "^", "cluster"),
           "nb3_ensemble": (3, "D", "3-body ensemble"), "nb2_kepler": (3, "P", "Kepler (2 bodies)")}
    for name, (i, mk, lab) in fam.items():
        g = d[(d.family == name) & (d.bits == 53)].sort_values("N")
        if len(g):
            axs[0].plot(g.N, g.tpred_at_max_n, color=t["CAT"][i], marker=mk, label=lab)
            axs[1].plot(g.N, g.n_star_empirical, color=t["CAT"][i], marker=mk)
    axs[0].set_xlabel("number of bodies N"); axs[0].set_ylabel("T_pred at top n, f64")
    axs[1].set_xlabel("number of bodies N"); axs[1].set_ylabel("empirical n*, f64")
    axs[0].legend(fontsize=8)
    fig.suptitle("Dimension scaling, N-body", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e11_dimension", theme)
    g = d[d.family == "lorenz96"]
    if len(g):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for b in sorted(g.bits.unique()):
            h = g[g.bits == b].sort_values("N")
            ax.plot(h.N, h.tpred_at_max_n, color=bits_color(t, b), marker=bits_marker(b), label=BITS_LABEL[b])
        ax.set_xlabel("N"); ax.set_ylabel("T_pred at top n"); ax.legend(fontsize=8); ax.set_title("Lorenz-96", fontsize=9.5)
        fig.tight_layout()
        save(fig, "e11_lorenz96", theme)


@both
def indicator(theme, t):
    d = read("e7_indicator_grid")
    if d is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, col, lab in ((axs[0], "ftle", "finite-time Lyapunov exponent"), (axs[1], "sens_n16", "log10 sensitivity to (n, p)")):
        sc = ax.scatter(d.y, d.py, c=d[col], s=14, cmap="Blues" if theme == "light" else "Blues_r", marker="s", lw=0)
        fig.colorbar(sc, ax=ax, label=lab)
        ax.set_xlabel("y"); ax.set_ylabel("p_y"); ax.grid(False)
    fig.suptitle("Chaos indicator on the Hénon–Heiles section, E = 1/8", color=t["INK"], fontsize=11)
    fig.tight_layout()
    save(fig, "e7_indicator", theme)


if __name__ == "__main__":
    stats_vs_traj(); escapes(); representation(); methods(); hardware(); spurious(); dimension(); indicator()
