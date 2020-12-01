import logging
import json
from pathlib import Path
from typing import Dict, Union
import time

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer

_log = logging.getLogger(__name__)


class RandomSearchOptimizer(SingleFidelityOptimizer):
    """
    This class implements a random search optimizer.
    All benchmark and optimizer specific information are stored in the dictionaries
    benchmark_settings and optimizer_settings.
    """
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        # Setup can be done here or in run()
        _log.info('Successfully initialized')

    def setup(self):
        pass

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """
        # Get the configuration space from the benchmark
        cs = self.benchmark.get_configuration_space(seed=self.rng)

        results = []
        num_configs = 1

        best_seen = 10000000
        best_conf = None
        # Run this forever, the bookkeeper benchmark will terminate this once it exceeds the limit
        while True:
            # Randomly sample a configuration
            _log.debug(f"Iteration: [{num_configs + 1}] Start sampling configurations")
            configuration = cs.sample_configuration()
            result = self.benchmark.objective_function(configuration, rng=self.rng,
                                                       **self.settings_for_sending)
            results.append((configuration, result))
            _log.info(f'Config [{num_configs:6d}] - Result: {result["function_value"]:.4f} - '
                      f'Time Used: {self.benchmark.get_total_time_used()}|'
                      f'{self.benchmark.wall_clock_limit_in_s}')
            num_configs += 1

            # From time to time we do some bookkeeping and store the incumbent ourselves
            # Note: For random search this is not necessary since the bookkeeper benchmark tracks
            # all information and we can re-compute the incumbent trajectory from the runhistory
            if result["function_value"] < best_seen:
                best_conf = dict(configuration)
                best_seen = result["function_value"]
            if (num_configs % 1) == 0:
                entry = {
                    "num_evals": num_configs,
                    "config": best_conf,
                    "timestamp": time.time(),
                }
                self.__save_results(entry)

    def __save_results(self, entry):
        with open(self.output_dir / "own_trajectory.json", "a") as fh:
            # Output entry as a json
            fh.write(json.dumps(entry) + "\n")
