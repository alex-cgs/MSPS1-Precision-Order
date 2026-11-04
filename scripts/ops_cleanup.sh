#!/bin/bash
# 04:28 safety net: nothing of ours may run on ops after 04:30 (spec §11); then a last rsync.
ROOT=/home/alexandre.tinouert/Desktop/Code/msps1
LOG=$ROOT/logs/ops_night_$(date +%F).log
exec >> "$LOG" 2>&1
echo "== cleanup $(date)"
ssh -o BatchMode=yes -o ConnectTimeout=20 ops 'pkill -u $USER -f "bench/e10_bench.py" ; sleep 1; pgrep -u $USER -f e10_bench -a || echo clean'
rsync -az -e "ssh -o BatchMode=yes" ops:msps1/results/raw/E10/ $ROOT/results/raw/E10/ || echo "rsync down failed"
