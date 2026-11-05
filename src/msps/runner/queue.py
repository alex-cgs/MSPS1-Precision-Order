"""Experiment YAML -> job grid -> idempotent SQLite queue (spec §9 rules 2, 5). run_id is a hash of the job config."""
from __future__ import annotations

import hashlib
import itertools
import json
import multiprocessing as mp
import os
import resource
import sqlite3
import time
import traceback
from multiprocessing.connection import wait as mp_wait
from pathlib import Path

import yaml

from .. import io as rio
from ..systems import load_card
from .provenance import provenance


def load_experiment(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _ic_ids(case_spec: str, mode: str) -> list:
    name = case_spec.split(":")[0]
    card = load_card(name)
    count = card.get("ic", {}).get("count")
    if count is None:
        return [0]
    return list(range(int(count[mode])))


def expand(exp: dict, mode: str) -> list:
    """List of (group_name, config). `ic: auto` expands per case from the card's dev or paper count."""
    out = []
    for group in exp["groups"]:
        if group.get("modes") and mode not in group["modes"]:
            continue
        over = group.get(mode, {}) or {}
        grid = {**group.get("grid", {}), **over.get("grid", {})}
        fixed = {**group.get("fixed", {}), **over.get("fixed", {})}
        keys = [k for k in grid if k != "ic"]
        for combo in itertools.product(*[grid[k] for k in keys]):
            cfg = dict(zip(keys, combo))
            ics = grid.get("ic", [0])
            for sub in group.get("explode", [None]):
                base = {**fixed, **cfg, **(sub or {})}
                for ic in (_ic_ids(base["case"], mode) if ics == "auto" else ics):
                    out.append((group["name"], {"kind": group["kind"], **base, "ic": ic}))
    # keep one IC's jobs together: workers cache a few references (stable sort, group order preserved)
    rank = {g["name"]: i for i, g in enumerate(exp["groups"])}
    out.sort(key=lambda j: (rank[j[0]], str(j[1].get("case", "")), j[1].get("ic", 0)))
    return out


def run_id(study: str, cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "max_wall"}
    return hashlib.sha1(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Queue:
    def __init__(self, study: str):
        d = rio.RESULTS / "queue"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{study}.sqlite"
        self.study = study
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("""create table if not exists jobs(run_id text primary key, grp text, seq int, config text, status text,
                           attempts int default 0, wall real, host text, started real, finished real, error text)""")
        self.db.commit()

    def add(self, jobs: list):
        for seq, (grp, cfg) in enumerate(jobs):
            rid = run_id(self.study, cfg)
            status = "done" if rio.run_done(self.study, rid) else "pending"
            self.db.execute("insert or ignore into jobs(run_id, grp, seq, config, status) values(?,?,?,?,?)",
                            (rid, grp, seq, json.dumps(cfg, default=str), status))
            if status == "done":
                self.db.execute("update jobs set status='done' where run_id=?", (rid,))
        self.db.execute("update jobs set status='pending' where status='running'")
        self.db.commit()

    def pending(self, grp: str) -> list:
        q = self.db.execute("select run_id, config from jobs where grp=? and status='pending' order by seq", (grp,))
        return [(r, json.loads(c)) for r, c in q.fetchall()]

    def mark(self, rid, status, wall=None, error=None):
        self.db.execute("update jobs set status=?, wall=?, error=?, attempts=attempts+1, finished=?, host=? where run_id=?",
                        (status, wall, error, time.time(), os.uname().nodename, rid))
        self.db.commit()

    def counts(self, ids=None) -> dict:
        """Status counts, restricted to `ids` (the current grid) when given; older rows stay in the table."""
        out = {}
        for rid, status in self.db.execute("select run_id, status from jobs").fetchall():
            if ids is None or rid in ids:
                out[status] = out.get(status, 0) + 1
        return out

    def reset_failed(self):
        self.db.execute("update jobs set status='pending' where status='failed'")
        self.db.commit()


def execute_job(args):
    """Worker body. Returns (run_id, status, wall, error). Writes the run directory itself."""
    study, rid, cfg = args
    from .jobs import KINDS
    if rio.run_done(study, rid):
        return rid, "done", 0.0, None
    t0 = time.time()
    try:
        cfg2 = {**cfg, "_study": study, "_run_id": rid}
        meta, data, states = KINDS[cfg["kind"]](cfg2)
        meta = {**meta, "config": cfg, "study": study, "run_id": rid, "clock_start": t0, "clock_end": time.time(),
                "wall_total": time.time() - t0, "provenance": provenance(cfg.get("ic"))}
        rio.write_run(study, rid, meta, data, states)
        return rid, "done", time.time() - t0, None
    except Exception:
        return rid, "failed", time.time() - t0, traceback.format_exc()[-1500:]


def worker_mem_bytes(workers: int, mem_gb: float | None = None) -> int:
    """Per-worker address-space cap: a runaway job fails on its own instead of taking the machine down."""
    if mem_gb:
        return int(mem_gb * 2**30)
    with open("/proc/meminfo") as f:
        avail = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable"))
    return int(0.8 * avail / workers)


def _init_worker(mem_bytes=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if mem_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


def _worker_loop(conn, mem_bytes, job_fn):
    _init_worker(mem_bytes)
    while (task := conn.recv()) is not None:
        conn.send(job_fn(task))


def run_pool(args, workers, mem_bytes=None, job_limit_s=None, job_fn=execute_job, max_tasks=200):
    """Yields (run_id, status, wall, error) per job. Unlike mp.Pool, whose imap waits forever for a job lost with its worker,
    a worker that dies (GMP and FLINT abort on a failed allocation) or passes job_limit_s fails that job and is replaced."""
    ctx = mp.get_context("spawn")
    todo = list(reversed(args))
    busy = {}  # worker conn -> [process, task, t_start, tasks_done]
    idle = []

    def spawn():
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker_loop, args=(child, mem_bytes, job_fn))
        proc.start()
        child.close()
        return parent, proc, 0

    def stop(conn, proc):
        proc.kill()
        proc.join()
        conn.close()

    try:
        while todo or busy:
            while todo and len(busy) < workers:
                conn, proc, n = idle.pop() if idle else spawn()
                task = todo.pop()
                conn.send(task)
                busy[conn] = [proc, task, time.time(), n]
            ready = set(mp_wait(list(busy) + [w[0].sentinel for w in busy.values()], timeout=2))
            for conn, (proc, task, t0, n) in list(busy.items()):
                wall = time.time() - t0
                try:
                    result = conn.recv() if conn in ready and conn.poll() else None
                except EOFError:
                    proc.join(5)
                    result = None
                reuse = result is not None and n + 1 < max_tasks
                if result is None:
                    if not proc.is_alive():
                        result = (task[1], "failed", wall, f"worker died, exit code {proc.exitcode}")
                    elif job_limit_s is not None and wall > job_limit_s:
                        result = (task[1], "failed", wall, f"killed at the job wall limit of {job_limit_s} s")
                    else:
                        continue
                del busy[conn]
                if reuse:
                    idle.append((conn, proc, n + 1))
                else:
                    stop(conn, proc)
                yield result
    finally:
        for conn, w in busy.items():
            stop(conn, w[0])
        for conn, proc, _ in idle:
            stop(conn, proc)


def run_experiment(path, mode="dev", workers=4, stop_after_s=None, limit=None, window=None, raw_budget_gb=20.0, log=print,
                   groups=None, mem_gb=None) -> dict:
    exp = load_experiment(path)
    study = exp["study"]
    q = Queue(study)
    jobs = expand(exp, mode)
    if groups:
        jobs = [j for j in jobs if j[0] in groups]
    q.add(jobs)
    ids = {run_id(study, c) for _, c in jobs}
    mem_bytes = worker_mem_bytes(workers, mem_gb)
    t_start = time.time()
    order = []
    for g, _ in jobs:
        if g not in order:
            order.append(g)
    log(f"{study} {mode}: {len(jobs)} jobs in {len(order)} groups {q.counts(ids)} worker_mem={mem_bytes / 2**30:.2f}GB")
    stopped = False
    for grp in order:
        pending = [(r, c) for r, c in q.pending(grp) if r in ids]
        if limit:
            pending = pending[:limit]
        if not pending:
            continue
        log(f"{study}/{grp}: {len(pending)} pending")
        args = [(study, r, c) for r, c in pending]
        done = 0
        results = run_pool(args, workers, mem_bytes, exp.get("job_wall_limit_s"))
        for rid, status, wall, err in results:
            q.mark(rid, status, wall, err)
            done += 1
            if err:
                log(f"FAILED {rid}: {err.splitlines()[-1] if err else ''}")
            if done % max(1, len(args) // 20) == 0:
                log(f"{study}/{grp}: {done}/{len(args)} elapsed {time.time() - t_start:.0f}s")
            over_budget = done % 200 == 0 and rio.disk_usage_mb(rio.RESULTS / "raw") / 1e3 > raw_budget_gb
            over_time = stop_after_s is not None and time.time() - t_start > stop_after_s
            over_window = window is not None and window.must_stop(margin_s=30)
            if over_budget or over_time or over_window:
                log(f"{study}: stopping (budget={over_budget} time={over_time} window={over_window})")
                results.close()
                stopped = True
                break
        if stopped:
            break
    wall = time.time() - t_start
    counts = q.counts(ids)
    log(f"{study} {mode}: finished in {wall:.0f}s {counts}")
    rec = rio.RESULTS / "raw" / study
    rec.mkdir(parents=True, exist_ok=True)
    with open(rec / "_campaign_log.jsonl", "a") as f:
        f.write(json.dumps({"mode": mode, "wall": wall, "counts": counts, "workers": workers, "t": t_start,
                            "budget_h": exp.get("budget_hours"), "stopped": stopped, "provenance": provenance()}, default=str) + "\n")
    return counts
