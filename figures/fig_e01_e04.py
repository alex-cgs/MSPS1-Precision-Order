import re

from style import BITS_LABEL, MARKERS, both, np, plt, read, save, short_case


@both
def local_law(theme, t):
    med, law = read("e1_local_medians"), read("e1_local_law")
    if med is None:
        return
    cases = sorted(med.case.unique(), key=lambda c: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", c)])
    cols = 4
    rows = int(np.ceil(len(cases) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.3 * rows), squeeze=False, sharey=True)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, c in zip(axs.flat, cases):
        ax.set_visible(True)
        d = med[med.case == c]
        r = law[law.case == c].iloc[0]
        ax.fill_between(d.n, d.q25, d.q75, color=t["BAND"], lw=0)
        ax.plot(d.n, d["median"], color=t["CAT"][0], marker="o", ms=3.5, label="median")
        ax.plot(d.n, r.log10_K + r.slope * (d.n + 1), color=t["INK2"], ls="--", lw=1, label="fit")
        ax.set_title(f"{short_case(c)}: ρ/h ≈ {r.rho_over_h:.1f}", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("Taylor order n")
    for row in axs:
        row[0].set_ylabel("log10 one-step error")
    fig.suptitle("Local truncation error per step against order (the schedule sets h between ρ/16 and ρ/8)", color=t["INK"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, "e1_local_law", theme)


@both
def exact_growth(theme, t):
    d = read("e4_exact_growth")
    if d is None:
        return
    d = d.dropna(subset=["growth_mean_after_first"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    fam = {"N-body": d[d.case.str.startswith("nb")], "Lorenz-63": d[d.case.str.startswith("lorenz")], "Hénon–Heiles": d[d.case.str.startswith("henon")], "Maps": d[d.case.str.contains("map")]}
    lo, hi = 1.2, max(d.predicted_factor.max(), d.growth_mean_after_first.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], color=t["MUTED"], ls="--", lw=1, label="measured = (d−1)n+1")
    for i, (name, g) in enumerate(fam.items()):
        ax.scatter(g.predicted_factor, g.growth_mean_after_first, s=38, color=t["CAT"][i], marker=MARKERS[i], edgecolor=t["SURF"], lw=0.8, label=name, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted growth factor per step, (d−1)n+1")
    ax.set_ylabel("measured growth factor per step")
    ax.set_title("Exact rationals: bit growth per step against (d−1)n+1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "e4_exact_growth", theme)


if __name__ == "__main__":
    local_law(); exact_growth()
