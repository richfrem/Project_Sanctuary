#!/bin/sh
# MISSION-074 held-out run (frozen). Seeds 7401 (p=0.5), 7402 (uniform 0.35), 7403 (checker 0.35).
set -e; mkdir -p heldout
n64=20000; n128=10000; n256=4000; n512=2000; n1024=1000
for L in 64 128 256 512 1024; do
  eval n=\$n$L
  ./tilings $L 0.5  $n 7401 > heldout/p050_L$L.jsonl
  ./tilings $L 0.35 $n 7402 > heldout/p035_L$L.jsonl
  ./tilings $L 0.35 $n 7403 checker > heldout/c035_L$L.jsonl
done
python3 analyze.py heldout > heldout_result.json
