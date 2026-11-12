#!/bin/bash
# MISSION-076 held-out run. Seeds 7600..; frozen before execution (see freeze.json). Do not edit after freeze.
set -e; cd "$(dirname "$0")"; mkdir -p heldout; PC=0.6447001; T=6000
jobs_file=$(mktemp)
echo "./sim --p $PC --phi 0 --runs 2000 --batches 20 --tmax $T --seed 7600 > heldout/base.csv" >> $jobs_file
for phi in 0.2 0.1 0.05 0.025 0.0125 0.00625; do
  echo "./sim --p $PC --phi $phi --runs 1500 --batches 20 --tmax $T --seed 7600 > heldout/lr_$phi.csv" >> $jobs_file
done
for phi in 0.2 0.1 0.05 0.025; do
  p=$(python3 -c "print($PC+$phi/2)")
  echo "./sim --p $p --phi 0 --runs 1500 --batches 20 --tmax $T --seed 7600 > heldout/nn_$phi.csv" >> $jobs_file
done
xargs -P 5 -I{} bash -c "{}" < $jobs_file
rm $jobs_file
python3 analyze.py heldout > heldout_result.json
python3 - <<'P'
import json,hashlib,glob
h={f:hashlib.sha256(open(f,'rb').read()).hexdigest() for f in sorted(glob.glob('heldout/*.csv'))}
json.dump(h,open('heldout_raw_sha256.json','w'),indent=1)
P
