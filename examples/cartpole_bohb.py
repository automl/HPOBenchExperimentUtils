"""
BOHB on Cartpole
================

This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.

Please install the necessary dependencies via ``pip install .[cartpole_example]``
and the HPOlib3 ``pip install <dir of hpolib>``
"""

import logging
logging.basicConfig(level=logging.DEBUG)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
logger = logging.getLogger('BOHB on cartpole')

from pathlib import Path

from HPOlibExperimentUtils import validate_benchmark

benchmark = 'xgboost'
output_dir = Path('../example_dir/xgboost')
rng = 1

# TODO
from hpolib.util.openml_data_manager import get_openmlcc18_taskids
task_id = get_openmlcc18_taskids()[0]
#
# run_benchmark(optimizer='bohb_eta_3_test',
#               benchmark=benchmark,
#               output_dir=output_dir,
#               rng=rng,
#               task_id=task_id,
#               use_local=False)

validate_benchmark(benchmark=benchmark,
                   output_dir=output_dir,
                   task_id=task_id,
                   rng=rng)
