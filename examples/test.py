import multiprocessing
from pathlib import Path
from time import time, sleep

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from hpolib.util.openml_data_manager import get_openmlcc18_taskids

from HPOlibExperimentUtils.optimizer.bohb_optimizer import BOHBOptimizer
from HPOlibExperimentUtils.utils.runner_utils import get_benchmark_settings, OptimizerEnum

task_id = get_openmlcc18_taskids()[0]
output_path = Path('./test')

benchmark = XGBoostBenchmark(task_id=task_id)
optimizer_settings, benchmark_settings = get_benchmark_settings(benchmark='xgboost', rng=0, output_dir=output_path)

optimizer = BOHBOptimizer(benchmark=benchmark,
                          optimizer_settings=optimizer_settings,
                          benchmark_settings=benchmark_settings,
                          intensifier=OptimizerEnum.BOHB,
                          rng=0)

# optimizer = SMACOptimizer(benchmark=benchmark,
#                           optimizer_settings=optimizer_settings,
#                           benchmark_settings=benchmark_settings,
#                           intensifier=OptimizerEnum.HYPERBAND,
#                           rng=0)


def target_fct() -> None:
    optimizer.run()


start_time = time()
process = multiprocessing.Process(target=target_fct, args=(), kwargs=dict())
process.start()

# Check if global time limit is reached or process is not alive anymore
while time() - start_time < 100000 and process.is_alive():
    sleep(0.5)

if process.is_alive():
    print('Terminate the still running process')
    process.terminate()

# process.join(5)  # enforce global time limit! :)
print(f'Finished after {time() - start_time:.2f}s')
