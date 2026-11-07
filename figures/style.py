"""Figure style: palette validated with the dataviz checker (light and dark). Figures read only results/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DERIVED = ROOT / "results" / "derived"
OUT = ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(INK="#0b0b0b", INK2="#52514e", MUTED="#898781", GRID="#e8e7e1", BASE="#c3c2b7", SURF="#fcfcfb",
                  CAT=["#2a78d6", "#eb6834", "#1baf7a", "#7a5cc0"],
                  RAMP=["#a9ccf5", "#7fb1ec", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"], BAND="#dbe8fa"),
    "dark": dict(INK="#f0efec", INK2="#c3c2b7", MUTED="#898781", GRID="#2c2c2a", BASE="#383835", SURF="#1a1a19",
                 CAT=["#3987e5", "#d95926", "#199e70", "#9b7fe0"],
                 RAMP=["#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cfe1f9"], BAND="#1c2f47"),
}
BITS_ORDER = [24, 53, 80, 106, 113, 160, 256]
BITS_LABEL = {24: "f32 (24)", 53: "f64 (53)", 64: "f80 (64)", 80: "80 bits", 106: "dd (106)", 113: "113 bits", 160: "160 bits", 256: "256 bits"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
DASHES = ["-", "--", ":", "-."]


def apply(theme: str) -> dict:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": t["BASE"], "axes.labelcolor": t["INK2"], "xtick.color": t["MUTED"], "ytick.color": t["MUTED"],
                         "axes.grid": True, "grid.color": t["GRID"], "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": t["SURF"], "axes.facecolor": t["SURF"], "savefig.facecolor": t["SURF"], "lines.linewidth": 1.6, "lines.markersize": 5,
                         "legend.frameon": False, "axes.titlecolor": t["INK"], "axes.titlesize": 10.5, "axes.titleweight": "regular", "text.color": t["INK2"], "legend.labelcolor": t["INK2"],
                         "legend.fontsize": 8.5, "pdf.fonttype": 42})
    return t


def bits_color(t, bits):
    return t["RAMP"][BITS_ORDER.index(bits)] if bits in BITS_ORDER else t["MUTED"]


def bits_marker(bits):
    return MARKERS[BITS_ORDER.index(bits) % len(MARKERS)] if bits in BITS_ORDER else "."


def save(fig, name: str, theme: str) -> None:
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / f"{name}_{theme}.pdf", bbox_inches="tight")
    plt.close(fig)


def read(name: str) -> pd.DataFrame | None:
    p = DERIVED / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def both(fn):
    """Decorator: draw in both themes."""
    def run():
        for theme in THEMES:
            fn(theme, apply(theme))
    run.__name__ = fn.__name__
    return run


def short_case(c: str) -> str:
    for old, new in (("nb2_kepler:e_9_10", "Kepler e=9/10"), ("nb2_kepler:e_1_2", "Kepler e=1/2"), ("nb2_kepler:e_0", "Kepler e=0"), ("nb3_figure_eight", "figure-eight"), ("nb3_lagrange", "Lagrange"), ("nb3_pythagorean", "Pythagorean"),
                     ("nbN_alternating:hier_N_", "hierarchical N="), ("nbN_alternating:rings_N_", "rings N="), ("nbN_circle:N_", "circle N="), ("nbN_cluster:N_", "cluster N="), ("_T8", ", T=8"), ("_hp", ", T=30")):
        c = c.replace(old, new)
    return c.replace(":r2", "").replace(":r4", "").replace("_r2", "").replace("henon_heiles:E_1_8", "Hénon–Heiles E=1/8").replace("henon_heiles:E_1_6", "Hénon–Heiles E=1/6").replace("lorenz63:rho_28", "Lorenz-63").replace("rossler", "Rössler") \
        .replace("nb3_pilot", "3-body pilot").replace("nb3_ensemble", "3-body ensemble").replace("double_pendulum", "double pendulum").replace("duffing:gamma_0_50", "Duffing γ=0.5") \
        .replace("lorenz96:N_", "Lorenz-96 N=").replace("fput_beta:N_", "FPUT-β N=")


__all__ = ["plt", "np", "pd", "both", "save", "read", "bits_color", "bits_marker", "BITS_LABEL", "BITS_ORDER", "MARKERS", "DASHES", "short_case", "DERIVED", "OUT", "ROOT"]
