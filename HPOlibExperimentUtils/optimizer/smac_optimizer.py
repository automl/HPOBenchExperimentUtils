import logging
from pathlib import Path
from time import time
from typing import Union, Dict, Type

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.hyperband import Hyperband
from smac.intensification.successive_halving import SuccessiveHalving
from smac.scenario.scenario import Scenario

from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from HPOlibExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOlibExperimentUtils.utils.optimizer_utils import get_number_ta_runs

logger = logging.getLogger('Optimizer')
# logging.basicConfig(level=logging.DEBUG)
root_logger = logging.getLogger()
# root_logger.setLevel(logging.DEBUG)


class SMACOptimizer(SingleFidelityOptimizer):
    """
    This class offers an interface to the SMAC Optimizer. It runs on a given benchmark.
    All benchmark and optimizer specific information are stored in the dictionaries benchmark_settings and
    optimizer_settings.
    The intensifier specifies which SMAC-Intensifier (HB or SH) is used.
    """
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 intensifier: Union[Type[Hyperband], Type[SuccessiveHalving]],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        self.intensifier = intensifier
        super().__init__(benchmark, settings, output_dir, rng)

    def setup(self):
        pass

    def run(self):
        """ Start the optimization run with SMAC (HB or SH). """
        number_ta_runs = get_number_ta_runs(iterations=self.settings['num_iterations'],
                                            min_budget=self.settings['min_budget'],
                                            max_budget=self.settings['max_budget'],
                                            eta=self.settings['eta'])

        scenario_dict = {"run_obj": "quality",
                         "cs": self.cs,
                         "deterministic": "true",
                         "limit_resources": False,
                         "output_dir": str(self.output_dir)}

        scenario = Scenario(scenario_dict)

        def optimization_function_wrapper(cfg, seed, instance, budget):
            """ Helper-function: simple wrapper to use the benchmark with smac"""

            fidelity = {self.main_fidelity.name: budget}
            result_dict = self.benchmark.objective_function(configuration=cfg,
                                                            fidelity=fidelity,
                                                            **self.settings_for_sending,
                                                            rng=seed)

            return result_dict['function_value']

        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(self.rng),
                        tae_runner=optimization_function_wrapper,
                        intensifier=self.intensifier,  # you can also change the intensifier to use like this!
                        intensifier_kwargs={'initial_budget': self.min_budget,
                                            'max_budget': self.max_budget,
                                            'eta': self.settings['eta']}
                        )

        start_time = time()
        try:
            smac.optimize()
        finally:
            incumbent = smac.solver.incumbent
        end_time = time()
        logger.info(f'Finished Optimization after {int(end_time - start_time):d}s. Incumbent is {incumbent}')


class SMACOptimizerHyperband(SMACOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark=benchmark, settings=settings, intensifier=Hyperband,
                         output_dir=output_dir, rng=rng)


class SMACOptimizerSuccessiveHalving(SMACOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark=benchmark, settings=settings, intensifier=SuccessiveHalving,
                         output_dir=output_dir, rng=rng)
