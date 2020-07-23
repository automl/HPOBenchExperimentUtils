import logging
from pathlib import Path
from time import time
from typing import Union, Dict

import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.hyperband import Hyperband
from smac.intensification.successive_halving import SuccessiveHalving
from smac.scenario.scenario import Scenario

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOlibExperimentUtils.utils.optimizer_utils import get_number_ta_runs, parse_fidelity_type
from HPOlibExperimentUtils.utils.runner_utils import OptimizerEnum

logger = logging.getLogger('Optimizer')


class SMACOptimizer(Optimizer):
    """
    This class offers an interface to the SMAC Optimizer. It runs on a given benchmark.
    All benchmark and optimizer specific information are stored in the dictionaries benchmark_settings and
    optimizer_settings.
    The intensifier specifies which SMAC-Intensifier (HB or SH) is used.
    """
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 optimizer_settings: Dict, benchmark_settings: Dict,
                 intensifier: OptimizerEnum, rng: Union[int, None] = 0):
        super().__init__(benchmark, optimizer_settings, benchmark_settings, intensifier, rng)

        if intensifier is OptimizerEnum.HYPERBAND:
            self.intensifier = Hyperband
        elif intensifier is OptimizerEnum.SUCCESSIVE_HALVING:
            self.intensifier = SuccessiveHalving
        else:
            raise ValueError('Currently no other intensifier is supported')

    def setup(self):
        pass

    def run(self) -> Path:
        """ Start the optimization run with SMAC (HB or SH). """
        number_ta_runs = get_number_ta_runs(iterations=self.optimizer_settings['num_iterations'],
                                            min_budget=self.optimizer_settings['min_budget'],
                                            max_budget=self.optimizer_settings['max_budget'],
                                            eta=self.optimizer_settings['eta'])

        scenario_dict = {"run_obj": "quality",
                         "wallclock-limit": self.optimizer_settings['time_limit_in_s'],
                         "cs": self.cs,
                         "deterministic": "true",
                         "limit_resources": True,
                         "runcount-limit": number_ta_runs,  # max. number of function evaluations
                         "cutoff": self.optimizer_settings['cutoff_in_s'],
                         "memory_limit": self.optimizer_settings['mem_limit_in_mb'],
                         "output_dir": str(self.optimizer_settings['output_dir']),
                         }

        scenario = Scenario(scenario_dict)

        def optimization_function_wrapper(cfg, seed, instance, budget):
            """ Helper-function: simple wrapper to use the benchmark with smac"""
            fidelity_type = parse_fidelity_type(self.benchmark_settings['fidelity_type'])
            fidelity = {self.benchmark_settings['fidelity_name']: fidelity_type(budget)}

            result_dict = self.benchmark.objective_function(configuration=cfg,
                                                            **fidelity,
                                                            **self.benchmark_settings_for_sending)
            return result_dict['function_value']

        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(self.rng),
                        tae_runner=optimization_function_wrapper,
                        intensifier=self.intensifier,  # you can also change the intensifier to use like this!
                        intensifier_kwargs={'initial_budget': self.optimizer_settings['min_budget'],
                                            'max_budget': self.optimizer_settings['max_budget'],
                                            'eta': self.optimizer_settings['eta']}
                        )

        start_time = time()
        try:
            smac.optimize()
        finally:
            incumbent = smac.solver.incumbent
        end_time = time()
        logger.info(f'Finished Optimization after {int(end_time - start_time):d}s. Incumbent is {incumbent}')

        # Since BOHB and SMAC write the output to different directories, return it here.
        return Path(smac.output_dir)
