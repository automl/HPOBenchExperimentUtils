import logging
from pathlib import Path

from HPOlibExperimentUtils.utils.optimizer_utils import prepare_dict_for_sending

logger = logging.getLogger('Optimizer')


class Optimizer:
    def __init__(self, benchmark, optimizer_settings, benchmark_settings, intensifier, rng=0):
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
        self.benchmark_settings_for_sending =  prepare_dict_for_sending(benchmark_settings)

    def setup(self):
        raise NotImplementedError()

    def run(self) -> Path:
        raise NotImplementedError()
