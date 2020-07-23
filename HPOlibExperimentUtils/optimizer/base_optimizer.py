import logging
from pathlib import Path
from typing import Union, Dict

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOlibExperimentUtils.utils.optimizer_utils import prepare_dict_for_sending
from HPOlibExperimentUtils.utils.runner_utils import OptimizerEnum

logger = logging.getLogger('Optimizer')


class Optimizer:
    """ Base class for the Optimizer classes for SMACOptimizer, BOHBOptimizer and DragonflyOptimizer """
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 optimizer_settings: Dict, benchmark_settings: Dict,
                 intensifier: OptimizerEnum, rng: Union[int, None] = 0):
        self.benchmark = benchmark
        self.cs = benchmark.get_configuration_space()
        self.rng = rng
        self.optimizer_settings = optimizer_settings
        self.benchmark_settings = benchmark_settings
        self.intensifier = intensifier

        # Since we use containerized benchmarks. Information are sent via json format to the container. Therefore each
        # entry in the benchmark settings dict must be json serializable.
        # Numpy arrays and Path-like objects aren't serializable. --> remove unserializable things. There is no
        # benchmark which takes as input an non-serializable parameter.
        # Please change the given parameter then or remove it from the benchmark settings dict beforehand.
        self.benchmark_settings_for_sending = prepare_dict_for_sending(benchmark_settings)

    def setup(self):
        raise NotImplementedError()

    def run(self) -> Path:
        raise NotImplementedError()
