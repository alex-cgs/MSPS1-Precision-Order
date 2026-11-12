#!/bin/sh
# Rebuild every study, table and figure from scratch (paper grids). Idempotent: finished runs are skipped, so it can be
# stopped and relaunched at any point. Expect about 430 core-hours plus about 4 GPU-hours; see README.md.
#
# usage: scripts/reproduce.sh [runs|e10|analysis|figures|all]   (default: all)
# env:   WORKERS (default 7), VENV (default .venv), VENV_GPU (default .venv-gpu)
cd "$(dirname "$0")/.." || exit 1
WORKERS=${WORKERS:-7}
VENV=${VENV:-.venv}
VENV_GPU=${VENV_GPU:-.venv-gpu}
STAGE=${1:-all}
mkdir -p logs

runs() {
  # Groups inside one file run in order (references first). e11_high_n.yaml holds the high-precision references that
  # e11_high_n_t8.yaml needs, so it comes first. A second pass with fewer workers and more memory retries jobs that hit the cap.
  for f in e00_validation e01_local e02_sweep e04_exact e05_stats e06_spurious e07_indicator e08_methods e09_representation \
           e11_dimension e11_high_n e11_high_n_t8; do
    nice -n 5 "$VENV/bin/msps" run "experiments/$f.yaml" --mode paper --workers "$WORKERS" --mem-gb 2.5 --log "logs/$f.log" >> "logs/$f.out" 2>&1
    nice -n 5 "$VENV/bin/msps" run "experiments/$f.yaml" --mode paper --retry-failed --workers 3 --mem-gb 6 --log "logs/$f.log" >> "logs/$f.out" 2>&1
  done
}

e10() {
  # Needs the GPU environment. Labels name the device (figures/fig_others.py HW_LABEL); change them on other hardware. The CPU run uses one numpy core and stops at batch 1e4 (1e5 is about 15 h).
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$VENV_GPU/bin/python" bench/e10_bench.py --label local_gtx1080ti --size full >> logs/e10_gpu.log 2>&1
  "$VENV/bin/python" bench/e10_bench.py --backend numpy --label local_cpu_numpy --size cpu --accuracy-orders 16 24 >> logs/e10_cpu.log 2>&1
  "$VENV_GPU/bin/python" scripts/e10_bitwise_diff.py 100 >> logs/e10_bitwise.log 2>&1
}

analysis() {
  # e03 reads the tables written by e02.
  for a in e00_report e01_local e02_sweep e03_model e04_exact e04_hh_degree e05_stats e06_spurious e07_indicator e08_methods \
           e09_representation e10_hardware e11_dimension; do
    "$VENV/bin/python" "analysis/$a.py" >> logs/analysis.log 2>&1 || echo "analysis $a failed, see logs/analysis.log"
  done
}

figures() {
  "$VENV/bin/python" figures/make_all.py
}

case "$STAGE" in
  runs) runs ;;
  e10) e10 ;;
  analysis) analysis ;;
  figures) figures ;;
  all) runs; e10; analysis; figures ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
