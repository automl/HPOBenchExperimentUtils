import logging
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

        while True:
            logger.debug("Start sampling configurations")
            configuration = cs.sample_configuration()
            result = self.benchmark.objective_function(configuration, rng=self.rng)
            results.append((configuration, result))
            logger.info(f'Config [{num_configs}:6d] - Result: {result["function_value"]}.')
            num_configs += 1
