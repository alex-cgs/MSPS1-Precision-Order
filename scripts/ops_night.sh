#!/bin/bash
# H100 night job, started by cron at 02:30 (spec §11). Never touches ops outside 02:30-04:30.
# Night plan comes from scripts/ops_plan.txt: first line "env_small" on night one, then "full".
set -u
ROOT=/home/alexandre.tinouert/Desktop/Code/msps1
LOG=$ROOT/logs/ops_night_$(date +%F).log
exec >> "$LOG" 2>&1
echo "== start $(date)"
hhmm=$(date +%H%M)
if [ "$hhmm" -lt 0230 ] || [ "$hhmm" -ge 0425 ]; then echo "outside window, abort"; exit 3; fi
PLAN=$(head -n1 $ROOT/scripts/ops_plan.txt)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 ops"
secs_to_deadline() { echo $(( $(date -d "today 04:25" +%s) - $(date +%s) )); }

$SSH 'mkdir -p ~/msps1' || { echo "ssh failed"; exit 4; }
rsync -az --exclude .venv --exclude .venv-gpu --exclude results/raw --exclude results/queue --exclude logs --exclude .git \
   -e "ssh -o BatchMode=yes" $ROOT/ ops:msps1/ || { echo "rsync up failed"; exit 5; }
$SSH 'cd ~/msps1 && mkdir -p logs results/raw && (nvidia-smi; python3 --version; uname -a; free -g | head -2; df -h . | tail -1; which conda mamba uv)' 
LEFT=$(secs_to_deadline)
$SSH "cd ~/msps1 && if [ ! -x .venv-gpu/bin/python ]; then timeout 900 python3 -m venv .venv-gpu && timeout 900 .venv-gpu/bin/pip install -q --upgrade pip && timeout 1200 .venv-gpu/bin/pip install -q 'jax[cuda12]' numpy scipy sympy mpmath gmpy2 python-flint pandas pyarrow pyyaml psutil pytest; fi; .venv-gpu/bin/python -c 'import jax;print(jax.__version__, jax.devices())'"
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
