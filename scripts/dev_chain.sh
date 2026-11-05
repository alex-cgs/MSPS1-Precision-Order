#!/bin/sh
# Remaining dev grids in priority order, at most 7 workers beside E1.
cd "$(dirname "$0")/.."
for s in e08_methods:E8 e06_spurious:E6 e07_indicator:E7; do
  nice -n 5 .venv/bin/msps run "experiments/${s%%:*}.yaml" --workers 7 --mem-gb 2.5 --log "logs/${s##*:}.log" > "logs/${s##*:}.out" 2>&1
done
