#!/usr/bin/env sh

echo "Run $OPTIMIZER"

cd HPOBenchExperimentUtils

python run_benchmark.py \
  --output_dir . \
  --optimizer $OPTIMIZER \
  --benchmark OptimizerTestBenchmark \
  --rng 1
