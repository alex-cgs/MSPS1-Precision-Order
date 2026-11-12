#!/bin/bash
# H100 night job, started by cron at 02:30 (spec §11). Never touches ops outside 02:30-04:30.
# usage: ops_night.sh [env_small|full|tiny]   (default env_small; use "full" once a small night has worked)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs"; LOG=$ROOT/logs/ops_night_$(date +%F).log
exec >> "$LOG" 2>&1
echo "== start $(date)"
hhmm=$(date +%H%M)
if [ "$hhmm" -lt 0230 ] || [ "$hhmm" -ge 0425 ]; then echo "outside window, abort"; exit 3; fi
PLAN=${1:-env_small}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 ops"
secs_to_deadline() { echo $(( $(date -d "today 04:25" +%s) - $(date +%s) )); }

# Light check first: nothing is copied or installed on a machine without a GPU.
if ! $SSH 'hostname; test -e /dev/nvidiactl && nvidia-smi'; then
  echo "no NVIDIA GPU visible on ops: E10 skipped tonight"; echo "== end $(date)"; exit 6
fi
$SSH 'mkdir -p ~/msps1' || { echo "ssh failed"; exit 4; }
rsync -az --exclude .venv --exclude .venv-gpu --exclude results/raw --exclude results/queue --exclude logs --exclude .git \
   --include 'results/references/ics/***' --exclude 'results/references/***' \
   -e "ssh -o BatchMode=yes" $ROOT/ ops:msps1/ || { echo "rsync up failed"; exit 5; }
$SSH 'cd ~/msps1 && mkdir -p logs results/raw && (nvidia-smi; python3 --version; uname -a; free -g | head -2; df -h . | tail -1; which conda mamba uv)' 
LEFT=$(secs_to_deadline)
$SSH "cd ~/msps1 && if ! .venv-gpu/bin/python -c 'import jax' 2>/dev/null; then timeout 900 python3 -m venv .venv-gpu && timeout 900 .venv-gpu/bin/pip install -q --upgrade pip && timeout 1200 .venv-gpu/bin/pip install -q 'jax[cuda12]' numpy scipy sympy mpmath gmpy2 python-flint pandas pyarrow pyyaml psutil pytest; fi; .venv-gpu/bin/python -c 'import jax;print(jax.__version__, jax.devices())'" || { echo "JAX not usable on ops"; echo "== end $(date)"; exit 7; }
case "$PLAN" in
  env_small) ARGS="--size small --steps 16" ;;
  full) ARGS="--size full" ;;
  *) ARGS="--size tiny" ;;
esac
LEFT=$(secs_to_deadline)
echo "plan=$PLAN seconds_left=$LEFT"
COMMIT=$(git -C $ROOT rev-parse HEAD)
$SSH "cd ~/msps1 && MSPS_GIT_COMMIT=$COMMIT MSPS_ROOT=\$PWD XLA_PYTHON_CLIENT_PREALLOCATE=false timeout --kill-after=30 $LEFT .venv-gpu/bin/python bench/e10_bench.py --label ops_h100 $ARGS --deadline-guard"
echo "bench exit=$?"
rsync -az -e "ssh -o BatchMode=yes" ops:msps1/results/raw/E10/ $ROOT/results/raw/E10/ || echo "rsync down failed"
$SSH 'pgrep -u $USER -f "msps1|e10_bench" -a || true'
echo "== end $(date)"
