import logging
from pathlib import Path
from time import time
from typing import Union, Dict, Type

import ConfigSpace as CS
import numpy as np

from smac.facade.smac_bohb_facade import BOHB4HPO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bo_facade import SMAC4BO
from smac.intensification.hyperband import Hyperband
from smac.intensification.successive_halving import SuccessiveHalving
from smac.scenario.scenario import Scenario

from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer

_log = logging.getLogger(__name__)


class SMACOptimizer(SingleFidelityOptimizer):
    """
    This class offers an interface to the SMAC Optimizer. It runs on a given benchmark.
    All benchmark and optimizer specific information are stored in the dictionaries
    benchmark_settings and optimizer_settings.
    The intensifier specifies which SMAC-Intensifier (HB or SH) is used.
    """
    def __init__(self, benchmark: Bookkeeper,
                 intensifier: Union[Type[Hyperband], Type[SuccessiveHalving], None],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        self.intensifier = intensifier
        super().__init__(benchmark, settings, output_dir, rng)

    def setup(self):
        pass

    def get_scenario(self):
        scenario_dict = {"run_obj": "quality",
                         "cs": self.cs,
                         "deterministic": "true",
                         "limit_resources": False,
                         "output_dir": str(self.output_dir)}

        return Scenario(scenario_dict)

    def _setupsmac(self, scenario, optimization_function_wrapper):
        smac = BOHB4HPO(scenario=scenario,
                        rng=np.random.RandomState(self.rng),
                        tae_runner=optimization_function_wrapper,
                        intensifier=self.intensifier,
                        intensifier_kwargs={'initial_budget': self.min_budget,
                                            'max_budget': self.max_budget,
                                            'eta': self.settings['eta'],
                                            }
                        )
        return smac

    def run(self):
        """ Start the optimization run with SMAC (HB or SH). """

        def optimization_function_wrapper(cfg, seed, instance, budget):
            """ Helper-function: simple wrapper to use the benchmark with smac"""
            run_id = SingleFidelityOptimizer._id_generator()

            if isinstance(self.main_fidelity, CS.hyperparameters.UniformIntegerHyperparameter) \
                    or isinstance(self.main_fidelity, CS.hyperparameters.NormalIntegerHyperparameter) \
                    or isinstance(self.main_fidelity.default_value, int):
                budget = int(budget)
            fidelity = {self.main_fidelity.name: budget}
            result_dict = self.benchmark.objective_function(configuration=cfg,
                                                            configuration_id=run_id,
                                                            fidelity=fidelity,
                                                            **self.settings_for_sending,
                                                            rng=seed)
            return result_dict['function_value']

        # Allow at most max_stages stages
        tmp = self.max_budget
        for i in range(self.settings.get('max_stages', 10)):
            tmp /= self.settings.get('eta', 3)
        if tmp > self.min_budget:
            self.min_budget = tmp

        scenario = self.get_scenario()
        smac = self._setupsmac(scenario, optimization_function_wrapper)

        start_time = time()
        try:
            smac.optimize()
        finally:
            incumbent = smac.solver.incumbent
        end_time = time()
        _log.info(f'Finished Optimization after {int(end_time - start_time):d}s. '
                  f'Incumbent is {incumbent}')


class SMACOptimizerHyperband(SMACOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(SMACOptimizerHyperband, self).__init__(benchmark=benchmark, settings=settings,
                                                     intensifier=Hyperband, output_dir=output_dir,
                                                     rng=rng)


class SMACOptimizerSuccessiveHalving(SMACOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(SMACOptimizerSuccessiveHalving, self).__init__(benchmark=benchmark, settings=settings,
                                                             intensifier=SuccessiveHalving,
                                                             output_dir=output_dir, rng=rng)


class SMACOptimizerHPO(SMACOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(SMACOptimizerHPO, self).__init__(benchmark=benchmark, settings=settings,
                                               intensifier=None, output_dir=output_dir, rng=rng)

    def _setupsmac(self, scenario, optimization_function_wrapper):
        smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(self.rng),
                        tae_runner=optimization_function_wrapper)
        return smac

    def run(self):
        """ Start the optimization run with SMAC. """

        def optimization_function_wrapper(cfg, seed):
            """ Helper-function: simple wrapper to use the benchmark with smac"""
            run_id = SingleFidelityOptimizer._id_generator()

            fidelity = {self.main_fidelity.name: self.max_budget}
            result_dict = self.benchmark.objective_function(configuration=cfg,
                                                            configuration_id=run_id,
                                                            fidelity=fidelity,
                                                            **self.settings_for_sending,
                                                            rng=seed)
            return result_dict['function_value']

        scenario = self.get_scenario()
        smac = self._setupsmac(scenario, optimization_function_wrapper)

        start_time = time()
        try:
            smac.optimize()
        finally:
            incumbent = smac.solver.incumbent
        end_time = time()
        _log.info(f'Finished Optimization after {int(end_time - start_time):d}s. '
                  f'Incumbent is {incumbent}')


class SMACOptimizerBO(SMACOptimizerHPO):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark=benchmark, settings=settings, output_dir=output_dir, rng=rng)

    def _setupsmac(self, scenario, optimization_function_wrapper):
        smac = SMAC4BO(scenario=scenario, rng=np.random.RandomState(self.rng),
                       tae_runner=optimization_function_wrapper)
        return smac
