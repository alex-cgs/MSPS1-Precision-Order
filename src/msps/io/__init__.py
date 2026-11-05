"""Result schema (spec §9): results/raw/<study>/<run_id>/{data.parquet, states.parquet, meta.json}."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from ..systems import ROOT

RESULTS = ROOT / "results"


def run_dir(study: str, run_id: str) -> Path:
    return RESULTS / "raw" / study / run_id


def run_done(study: str, run_id: str) -> bool:
    return (run_dir(study, run_id) / "meta.json").exists()


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, default=_json_default, indent=1))
    tmp.replace(path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def write_table(path: Path, columns: dict) -> None:
    table = pa.table({k: pa.array(np.asarray(v)) for k, v in columns.items()})
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def write_run(study: str, run_id: str, meta: dict, data: dict | None = None, states: dict | None = None) -> Path:
    """meta.json is written last, so its presence marks a finished, complete run."""
    d = run_dir(study, run_id)
    d.mkdir(parents=True, exist_ok=True)
    if data:
        write_table(d / "data.parquet", data)
    if states:
        write_table(d / "states.parquet", states)
    _atomic_json(d / "meta.json", meta)
    return d


def read_meta(path) -> dict:
    return json.loads(Path(path).read_text())


def superseded_cases() -> set:
    """Cases whose stored runs must not be analysed; they stay on disk (experiments/superseded.yaml gives reasons)."""
    path = RESULTS.parent / "experiments" / "superseded.yaml"
    return set(yaml.safe_load(path.read_text())["cases"]) if path.exists() else set()


def iter_runs(study: str, include_superseded: bool = False):
    base = RESULTS / "raw" / study
    skip = set() if include_superseded else superseded_cases()
    if base.exists():
        for d in sorted(base.iterdir()):
            m = d / "meta.json"
            if m.exists():
                meta = json.loads(m.read_text())
                if meta.get("config", {}).get("case") not in skip:
                    yield d, meta


def load_data(run_path) -> pd.DataFrame:
    return pd.read_parquet(Path(run_path) / "data.parquet")


def load_states(run_path) -> pd.DataFrame:
    return pd.read_parquet(Path(run_path) / "states.parquet")


def disk_usage_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e6
