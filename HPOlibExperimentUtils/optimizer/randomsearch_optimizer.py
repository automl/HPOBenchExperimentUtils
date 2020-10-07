import logging
import pickle
from pathlib import Path
from typing import Dict, Union

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer

logger = logging.getLogger('RandomSearchOptimizer')


class RandomSearchOptimizer(Optimizer):
    """
    This class offers an interface to the BOHB Optimizer. It runs on a given benchmark.
    All benchmark and optimizer specific information are stored in the dictionaries benchmark_settings and
    optimizer_settings.
    """
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        logger.info('Successfully initialized')

    def setup(self):
        pass

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """

        cs = self.benchmark.get_configuration_space(seed=self.rng)

        results = []
        num_configs = 1
        num_iterations = self.settings['num_iterations']

        for iteration in range(num_iterations):
            logger.debug(f"Iteration: [{iteration + 1}|{num_iterations}] Start sampling configurations")
            configuration = cs.sample_configuration()
            result = self.benchmark.objective_function(configuration, rng=self.rng, **self.settings_for_sending)
            results.append((configuration, result))
            logger.info(f'Config [{num_configs:6d}] - Result: {result["function_value"]:.4f} - '
                        f'Time Used: {result["cost"]}')
            num_configs += 1

            if (num_configs % 100) == 0:
                self.__save_results(results)

        self.__save_results(results)

    def __save_results(self, results):
        with (self.output_dir / 'runhistory.pkl').open('wb') as fh:
            pickle.dump(results, fh)
