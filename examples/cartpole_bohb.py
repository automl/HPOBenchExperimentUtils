"""
BOHB on Cartpole
================

This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.

Please install the necessary dependencies via ``pip install .[cartpole_example]``
and the HPOlib3 ``pip install <dir of hpolib>``
"""

import logging
from pathlib import Path

from HPOlibExperimentUtils import OptimizerEnum, run_benchmark, validate_benchmark

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('BOHB on cartpole')
root_logger = logging.getLogger()

root_logger.setLevel(logging.DEBUG)

benchmark = 'cartpolereduced'
output_dir = Path('../example_dir/cartpole_bohb2')
rng = 1

run_benchmark(optimizer=OptimizerEnum.BOHB,
              benchmark=benchmark,
              output_dir=output_dir,
              rng=rng)

validate_benchmark(benchmark=benchmark,
                   output_dir=output_dir,
                   rng=rng)
