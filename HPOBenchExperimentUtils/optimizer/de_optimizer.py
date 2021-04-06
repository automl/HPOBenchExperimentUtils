import logging
from pathlib import Path
from typing import Union, Dict
import sys
import numpy as np
import json
import ConfigSpace as CS


from HPOBenchExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from dehb.optimizers import DE
from ConfigSpace import UniformFloatHyperparameter, Configuration

_log = logging.getLogger(__name__)


# Note that the DE Optimizer interface here assumes a single-fidelity case, or rather does not acknowledge the
# possible existence of multiple fidelity values.

class DEOptimizer(Optimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

        # Wrapper around the benchmark objective function to be passed to DE
        def f(config: Configuration):
            nonlocal self
            res = benchmark.objective_function(config)
            fitness, cost = res['function_value'], res['cost']
            return fitness, cost

        self.settings["verbose"] = _log.level <= logging.INFO
        # Set the number of iterations to a _very_ large integer but leave out some scope
        self.settings["gen"] = sys.maxsize >> 2

        # Parameter space to be used by DE
        cs = self.benchmark.get_configuration_space()
        dimensions = len(cs.get_hyperparameters())
        # Cross over, mutation 0.5, max_age=infinite, strategy=rand1_bin, pop_size=40
        # Initializing DE object
        self.de = DE(cs=cs, f=f, dimensions=dimensions, pop_size=self.settings["pop_size"],
                     mutation_factor=self.settings["mutation_factor"], crossover_prob=self.settings["crossover_prob"],
                     strategy=self.settings["strategy"])

    def setup(self):
        pass

    def run(self):
        np.random.seed(self.rng)
        # Running DE iterations
        # try:
        traj, runtime, history = self.de.run(generations=self.settings["gen"], verbose=self.settings["verbose"] or _log.level <= logging.DEBUG)
        # except TypeError as e:
        #     # The interface has changed for the DEHB optimizer. The new version has brackets instead of iterations.
        #     traj, runtime, history = self.de.run(brackets=self.settings["iter"],
        #                                          verbose=self.settings["verbose"] or _log.level <= logging.DEBUG)
