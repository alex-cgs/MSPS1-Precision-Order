#!/bin/sh
# Continues scripts/paper_chain.sh from E5 after the YAML changes of 2026-11-06. Idempotent.
cd "$(dirname "$0")/.." || exit 1
for s in e05_stats:E5 e09_representation:E9 e08_methods:E8 e06_spurious:E6 e07_indicator:E7 e04_exact:E4; do
  f=${s%%:*}; k=${s##*:}
  nice -n 5 .venv/bin/msps run "experiments/$f.yaml" --mode paper --workers 7 --mem-gb 2.5 --log "logs/$k.log" >> "logs/$k.out" 2>&1
  nice -n 5 .venv/bin/msps run "experiments/$f.yaml" --mode paper --retry-failed --workers 3 --mem-gb 6 --log "logs/$k.log" >> "logs/$k.out" 2>&1
done
