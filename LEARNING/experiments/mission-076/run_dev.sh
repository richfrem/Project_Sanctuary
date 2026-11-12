#!/bin/bash
# development pilot: dev seeds only (seed base 100); tmax 3000, 20 batches x 2000 runs
set -e; cd "$(dirname "$0")"; mkdir -p dev; PC=0.6447001
./sim --p $PC --phi 0 --runs 2000 --batches 20 --tmax 3000 --seed 100 > dev/base.csv
for phi in 0.1 0.03 0.01 0.003 0.001; do ./sim --p $PC --phi $phi --runs 2000 --batches 20 --tmax 3000 --seed 100 > dev/lr_$phi.csv; done
for phi in 0.1 0.03 0.01; do p=$(python3 -c "print($PC+$phi/2)"); ./sim --p $p --phi 0 --runs 2000 --batches 20 --tmax 3000 --seed 100 > dev/nn_$phi.csv; done
