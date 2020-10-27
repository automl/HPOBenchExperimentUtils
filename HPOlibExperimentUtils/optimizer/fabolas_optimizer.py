import logging
from pathlib import Path
from typing import Union, Dict
import sys

from HPOlibExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOlibExperimentUtils.optimizer.fabolas import fmin_fabolas
from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient

import ConfigSpace as cs
from emukit.core import ParameterSpace, ContinuousParameter

_log = logging.getLogger(__name__)

class FabolasOptimizer(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        super().__init__(benchmark, settings, output_dir, rng)
        self.space = generate_space(benchmark.get_configuration_space())
        if isinstance(self.main_fidelity, cs.UniformIntegerHyperparameter):
            self.s_min = max(self.main_fidelity.lower, 1)
            self.s_max = self.main_fidelity.upper
        elif isinstance(self.main_fidelity, cs.UniformFloatHyperparameter):
            raise NotImplementedError("This is still under construction")

        def wrapper(x, s):
            _log.debug("Calling SVM objective function with configuration %s and dataset size %.2f/%.2f." %
                       (x, s, self.s_max))
            res = benchmark.objective_function(x, fidelity={self.main_fidelity.name: s})
            return res["function_value"], res["cost"]

        self.benchmark_caller = wrapper
        try:
            self.n_init = self.settings.get("num_init_evals")
        except KeyError as e:
            raise KeyError("The optimizer settings must include the key 'num_init_evals'.") from e


    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting FABOLAS optimizer.")
        res = fmin_fabolas(func=self.benchmark_caller, space=self.space, s_min=self.s_min, s_max=self.s_max,
                           n_iters=sys.maxsize, n_init=self.n_init, marginalize_hypers=False)
        _log.info("FABOLAS optimizer finished.")

def generate_space(cspace: cs.ConfigurationSpace):
    """ Map a ConfigSpace.ConfigurationSpace object to an emukit compatible version. Only works for Continuous
    Hyperparameter Spaces."""

    space = []
    for parameter in cspace.get_hyperparameters():
        assert isinstance(parameter, cs.UniformFloatHyperparameter), \
            "FABOLAS only supports benchmarks with continuous parameter spaces, but parameter %s is of type %s." % \
            (parameter.name, str(type(parameter)))
        space.append(ContinuousParameter(parameter.name, parameter.lower, parameter.upper))

    return ParameterSpace(space)