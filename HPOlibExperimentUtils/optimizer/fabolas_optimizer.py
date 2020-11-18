import logging
from pathlib import Path
from typing import Union, Dict, Callable, Tuple, Sequence
import sys
from math import log, exp

from HPOlibExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from emukit.examples.fabolas import fmin_fabolas
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
        self.original_space = self.benchmark.get_configuration_space()
        self.emukit_space, self.to_emu, self.to_cs = _generate_space_mappings(self.original_space)
        if isinstance(self.main_fidelity, cs.UniformIntegerHyperparameter):
            _log.debug("Treating integer fidelity parameter %s as the main fidelity used for dataset subsampling." %
                       self.main_fidelity.name)
            self.s_min = max(self.main_fidelity.lower, 1)
            self.s_max = self.main_fidelity.upper
            self.subsample_to_cs_fidel = lambda x: x
        elif isinstance(self.main_fidelity, cs.UniformFloatHyperparameter):
            _log.debug("Treating float fidelity parameter %s as the main fidelity used for dataset subsampling." %
                       self.main_fidelity.name)
            assert hasattr(benchmark.benchmark, 'X_train'), "The benchmark object is expected to have an attribute " \
                                                            "'X_train' in order to be compatible with FABOLAS."
            assert hasattr(benchmark.benchmark, 'y_train'), "The benchmark object is expected to have an attribute " \
                                                            "'y_train' in order to be compatible with FABOLAS."
            assert 0.0 <= self.main_fidelity.lower and self.main_fidelity.upper <= 1.0
            self.s_min = max(self.main_fidelity.lower * benchmark.benchmark.y_train.shape[0], 1)
            self.s_max = self.main_fidelity.upper * benchmark.benchmark.y_train.shape[0]
            self.subsample_to_cs_fidel = lambda x: x / self.s_max
        else:
            raise RuntimeError("The benchmark's main fidelity parameter must be either a float or int, found "
                               "type %s" % type(self.main_fidelity))

        def wrapper(x, s):
            _log.debug("Calling objective function with configuration %s and dataset size %.2f/%.2f." %
                       (x, s, self.s_max))
            x = cs.Configuration(self.original_space, values={name: func(i) for (name, func), i in zip(self.to_cs, x)})
            res = benchmark.objective_function(x, fidelity={self.main_fidelity.name: self.subsample_to_cs_fidel(s)})
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
        res = fmin_fabolas(func=self.benchmark_caller, space=self.emukit_space, s_min=self.s_min, s_max=self.s_max,
                           n_iters=sys.maxsize, n_init=self.n_init,
                           marginalize_hypers=self.settings["marginalize_hypers"])
        _log.info("FABOLAS optimizer finished.")
        return self.output_dir


def _handle_uniform_float(param: cs.UniformFloatHyperparameter) -> Tuple[ContinuousParameter, Callable, Callable]:
    """ Generate a mapping for a UniformFloatHyperparameter object. """
    min_val, max_val = param.lower, param.upper
    if param.log:
        min_val = log(min_val)
        max_val = log(max_val)
        map_to_emu = lambda x: log(x)
        map_to_cs = lambda x: exp(x)
    else:
        map_to_emu = lambda x: x
        map_to_cs = lambda x: x
    emukit_param =  ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)

    return emukit_param, map_to_emu, map_to_cs


def _handle_uniform_int(param: cs.UniformIntegerHyperparameter) -> \
        Tuple[ContinuousParameter, Callable, Callable]:
    """ Generate a mapping for a UniformIntegerHyperparameter object. """

    if param.log:
        min_val, max_val = log(param.lower), log(param.upper)
        map_to_emu = lambda x: log(x)
        map_to_cs = lambda x: round(exp(x))
    else:
        min_val, max_val = param.lower, param.upper
        map_to_emu = lambda x: x
        map_to_cs = lambda x: round(x)

    emukit_param = ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)
    return emukit_param, map_to_emu, map_to_cs


param_map = {
    cs.UniformFloatHyperparameter: _handle_uniform_float,
    cs.UniformIntegerHyperparameter: _handle_uniform_int
}


def _generate_space_mappings(cspace: cs.ConfigurationSpace) -> \
        Tuple[ParameterSpace, Sequence[Tuple[str, Callable]], Sequence[Tuple[str, Callable]]]:
    """ Map a ConfigSpace.ConfigurationSpace object to an emukit compatible version and generate the relevant mappings
    to work across the two spaces. """

    space = []
    to_emu = []
    to_cs = []
    for parameter in cspace.get_hyperparameters():
        emukit_param, map_to_emu, map_to_cs = param_map[type(parameter)](parameter)
        space.append(emukit_param)
        to_emu.append((parameter.name, map_to_emu))
        to_cs.append((parameter.name, map_to_cs))

    return ParameterSpace(space), to_emu, to_cs