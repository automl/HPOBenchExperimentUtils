"""
BOHB on Cartpole
================

This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.

Please install the necessary dependencies via ``pip install .[cartpole_example]``
and the HPOBench ``pip install <dir of hpobench>``
"""

import logging
logger = logging.getLogger('BOHB on cartpole')
logger.setLevel(level=logging.DEBUG)

from pathlib import Path
from HPOlibExperimentUtils import validate_benchmark, run_benchmark

benchmark = 'cartpolereduced'
output_dir = Path('../example_dir/cartpole')
rng = 1

run_benchmark(optimizer='hpbandster_rs_eta_3_test',
              benchmark=benchmark,
              output_dir=output_dir,
              rng=rng,
              use_local=False)

validate_benchmark(benchmark=benchmark,
                   output_dir=output_dir,
                   rng=rng)
