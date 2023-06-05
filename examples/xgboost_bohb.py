"""
BOHB on XGBoost
================

This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.

Please install the necessary dependencies via ``pip install .[xgboost]``
and the HPOBench ``pip install <dir of hpobench>``
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BOHB on xgboost')

from pathlib import Path
from HPOBenchExperimentUtils import validate_benchmark, run_benchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids


benchmark = 'xgboostsub'
output_dir = Path('example_dir/xgboost')
rng = 1
task_id = get_openmlcc18_taskids()[0]

# Run benchmark stores the results in the defined output directory / hpbandster_bohb_eta_3_test-run-<seed=1>
# Additional parameters such as the taskid for the xgboost benchmark can simply added to the run-benchmark call.
# The script parses them automatically and forwards them to the benchmark.
# Each call of run_benchmark creates an own trajectory of the hpo run and runhistory with all function calls.

# Each configuration has a predefined cutoff time limit. And each optimization run has a global time limit.
# Both timelimits are definied in the benchmark_settings.
run_benchmark(optimizer='smac_sf',
              benchmark=benchmark,
              output_dir=output_dir,
              rng=rng,
              task_id=task_id,
              use_local=False)

# The validation script reads in all the found trajectory files in the defined outputdir and evalutes the
# configurations of all trajectories on the maximum budget.
validate_benchmark(benchmark=benchmark,
                   output_dir=output_dir.parent,
                   task_id=task_id,
                   rng=rng)
