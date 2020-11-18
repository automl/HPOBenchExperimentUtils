import logging
from abc import ABC
from pathlib import Path
from typing import Union, Dict

from ConfigSpace.hyperparameters import OrdinalHyperparameter, UniformIntegerHyperparameter
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from HPOlibExperimentUtils.utils.optimizer_utils import prepare_dict_for_sending, get_main_fidelity

_log = logging.getLogger(__name__)


class Optimizer(ABC):
    """ Base class for the Optimizer classes for SMACOptimizer, BOHBOptimizer and DragonflyOptimizer """
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        self.benchmark = benchmark
        self.cs = benchmark.get_configuration_space()
        self.rng = rng
        self.output_dir = output_dir
        self.settings = settings

        # Since we use containerized benchmarks. Information are sent via json format to the container. Therefore each
        # entry in the benchmark settings dict must be json serializable.
        # Numpy arrays and Path-like objects aren't serializable. --> remove unserializable things. There is no
        # benchmark which takes as input an non-serializable parameter.
        # Please change the given parameter then or remove it from the benchmark settings dict beforehand.
        self.settings_for_sending = prepare_dict_for_sending(settings)

    def setup(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class SingleFidelityOptimizer(Optimizer, ABC):

    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        # determine min and max budget from the fidelity space
        fidelity_space = benchmark.get_fidelity_space()
        self.main_fidelity = get_main_fidelity(fidelity_space, settings)

        if isinstance(self.main_fidelity, OrdinalHyperparameter):
            order = self.main_fidelity.get_seq_order()
            self.min_budget = self.main_fidelity.get_value(order.min())
            self.max_budget = self.main_fidelity.get_value(order.max())
        else:
            self.min_budget = self.main_fidelity.lower
            self.max_budget = self.main_fidelity.upper

        if isinstance(self.main_fidelity, UniformIntegerHyperparameter):
            # TODO: We need to fix this through never setting an int budget to 0
            self.min_budget = max(self.min_budget, 1)
            self.max_budget = max(self.max_budget, 1)
        else:
            self.min_budget = max(self.min_budget, 0.01)
            self.max_budget = max(self.max_budget, 0.01)

        super(SingleFidelityOptimizer, self).__init__(benchmark, settings, output_dir, rng)
