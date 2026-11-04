"""Shared loaders. Analysis reads results/raw and references, writes results/derived only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msps.io import RESULTS, iter_runs, load_data  # noqa: E402

DERIVED = RESULTS / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


def bits_of(arith: str) -> int:
    table = {"f16": 11, "f32": 24, "f64": 53, "f80": 64, "dd": 106}
    if arith in table:
        return table[arith]
    return int(arith.split("-")[1])


def save_table(df: pd.DataFrame, name: str, md_rows: int | None = None) -> None:
    """CSV plus a Markdown copy (AGENT_PROMPT section 9)."""
    df.to_csv(DERIVED / f"{name}.csv", index=False)
    show = df if md_rows is None else df.head(md_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in show.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    if md_rows is not None and len(df) > md_rows:
        lines.append(f"\n(first {md_rows} of {len(df)} rows; full table in {name}.csv)")
    (DERIVED / f"{name}.md").write_text("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return str(v)
        return f"{v:.4g}"
    return str(v)


def traj_rows(study: str, groups=None) -> pd.DataFrame:
    """One row per (run, tolerance). Keeps run_path for later E(t) reads."""
    rows = []
    for d, m in iter_runs(study):
        c = m.get("config", {})
        if c.get("kind") != "traj":
            continue
        for tol, r in m["tpred"].items():
            rows.append({"run": str(d), "case": c["case"], "ic": c["ic"], "arith": c["arith"], "bits": bits_of(c["arith"]),
                         "method": c.get("method"), "n": c.get("n"), "schedule": c.get("schedule"), "tol": float(tol), "tpred": r["tpred"],
                         "censored": bool(r["censored"]), "status": m["status"], "wall": m["wall"], "steps": m["steps"], "t_end": m["t_end"],
                         "substeps": c.get("substeps", 1), "s": c.get("s"), "k": c.get("k"), "rtol": c.get("rtol"), "solver_tol": c.get("solver_tol"),
                         "solver_iterations": m.get("solver_iterations"), "solver_failures": m.get("solver_failures"),
                         "sample_all": bool(c.get("sample_all", False))})
    return pd.DataFrame(rows)


def kind_runs(study: str, kind: str):
    for d, m in iter_runs(study):
        if m.get("config", {}).get("kind") == kind:
            yield d, m


def amp_table(study: str) -> dict:
    """{(case, ic): DataFrame t, A, log_growth, h, rho}"""
    out = {}
    for d, m in kind_runs(study, "amplification"):
        out[(m["config"]["case"], m["config"]["ic"])] = load_data(d)
    return out


def local_table(study: str) -> pd.DataFrame:
    rows = []
    for d, m in kind_runs(study, "local"):
        df = load_data(d)
        df["case"] = m["config"]["case"]
        df["ic"] = m["config"]["ic"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def iqr(a):
    return np.percentile(a, 25), np.percentile(a, 75)
