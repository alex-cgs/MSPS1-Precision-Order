"""msps run|dryrun|status|check-window. Output stays short; progress goes to the given log file."""
from __future__ import annotations

import argparse
import json
import random
import time

from .. import io as rio
from .queue import Queue, execute_job, expand, load_experiment, run_experiment, run_id


def _log_factory(path):
    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        if path:
            with open(path, "a") as f:
                f.write(line + "\n")
        print(line, flush=True)
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(prog="msps")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment")
    r.add_argument("--mode", default="dev", choices=["dev", "paper"])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--stop-after", type=float)
    r.add_argument("--limit", type=int)
    r.add_argument("--window", action="store_true", help="enforce the ops 02:30-04:25 guard")
    r.add_argument("--raw-budget-gb", type=float, default=20.0)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--groups", nargs="*")
    r.add_argument("--mem-gb", type=float, help="per-worker memory cap; default 80%% of available memory / workers")
    r.add_argument("--log")
    d = sub.add_parser("dryrun")
    d.add_argument("experiment")
    d.add_argument("--mode", default="dev")
    d.add_argument("--samples", type=int, default=2)
    d.add_argument("--workers", type=int, default=10)
    d.add_argument("--groups", nargs="*")
    s = sub.add_parser("status")
    s.add_argument("experiment")
    s.add_argument("--mode", default="dev", choices=["dev", "paper"])
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .window import Window
        win = None
        if a.window:
            win = Window()
            if not win.may_start(60):
                print("window guard: refusing to start")
                return 3
        if a.retry_failed:
            Queue(load_experiment(a.experiment)["study"]).reset_failed()
        run_experiment(a.experiment, a.mode, a.workers, a.stop_after, a.limit, win, a.raw_budget_gb, _log_factory(a.log), a.groups, a.mem_gb)
    elif a.cmd == "dryrun":
        exp = load_experiment(a.experiment)
        jobs = expand(exp, a.mode)
        if a.groups:
            jobs = [j for j in jobs if j[0] in a.groups]
        groups = {}
        for g, c in jobs:
            groups.setdefault(g, []).append(c)
        total_t, total_mb = 0.0, 0.0
        report = {}
        for g, cfgs in groups.items():
            rng = random.Random(0)
            cand = sorted(rng.sample(cfgs, min(a.samples, len(cfgs))), key=str)
            if len(cfgs) > 2 and cfgs[-1] not in cand:
                cand[-1] = cfgs[-1]
            ws, mbs = [], []
            for c in cand:
                rid = run_id(exp["study"], c)
                _, st, wall, err = execute_job((exp["study"] + "_dry", rid, c))
                if err:
                    print(err)
                ws.append(wall)
                mbs.append(rio.disk_usage_mb(rio.run_dir(exp["study"] + "_dry", rid)))
            est = sum(ws) / max(1, len(ws)) * len(cfgs)
            mb = sum(mbs) / max(1, len(mbs)) * len(cfgs)
            report[g] = {"jobs": len(cfgs), "sample_wall_s": [round(w, 2) for w in ws], "est_core_h": round(est / 3600, 2),
                         "est_wall_h": round(est / 3600 / a.workers, 2), "est_MB": round(mb, 1)}
            total_t += est
            total_mb += mb
        print(json.dumps({"study": exp["study"], "mode": a.mode, "groups": report,
                          "total_core_h": round(total_t / 3600, 2), "total_MB": round(total_mb, 1)}, indent=1))
    elif a.cmd == "status":
        exp = load_experiment(a.experiment)
        ids = {run_id(exp["study"], c) for _, c in expand(exp, a.mode)}
        print(exp["study"], a.mode, Queue(exp["study"]).counts(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
