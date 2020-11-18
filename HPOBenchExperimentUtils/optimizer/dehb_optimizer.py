import logging
from pathlib import Path
from typing import Union, Dict
import sys
import numpy as np
import json

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from dehb.optimizers import DE, DEHB
from ConfigSpace import UniformFloatHyperparameter, Configuration

_log = logging.getLogger(__name__)


class DehbOptimizer(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

        assert isinstance(self.main_fidelity, UniformFloatHyperparameter), \
            "DEHB only supports UniformFloat hyperparameters as main fidelity, received %s " % self.main_fidelity

        # Common objective function for DE & DEHB representing the benchmark
        def f(config: Configuration, budget=None):
            nonlocal self
            if budget is not None:
                res = benchmark.objective_function(config, fidelity={self.main_fidelity.name: budget})
            else:
                res = benchmark.objective_function(config)
            fitness, cost = res['function_value'], res['cost']
            return fitness, cost

        self.settings["verbose"] = _log.level <= logging.INFO
        # Set the number of iterations to a _very_ large integer but leave out some scope
        self.settings["iter"] = sys.maxsize >> 2

        # Parameter space to be used by DE
        cs = self.benchmark.get_configuration_space()
        dimensions = len(cs.get_hyperparameters())

        # Initializing DEHB object
        self.dehb = DEHB(cs=cs, dimensions=dimensions, f=f, strategy=self.settings["strategy"],
                    mutation_factor=self.settings["mutation_factor"], crossover_prob=self.settings["crossover_prob"],
                    eta=self.settings["eta"], min_budget=self.min_budget, max_budget=self.max_budget,
                    generations=self.settings["gens"], async_strategy=self.settings["async_strategy"])

    def setup(self):
        pass

    def run(self) -> Path:
        np.random.seed(self.rng)
        # Running DE iterations
        traj, runtime, history = self.dehb.run(iterations=self.settings["iter"], verbose=self.settings["verbose"],
                                               debug=_log.level <= logging.DEBUG)
