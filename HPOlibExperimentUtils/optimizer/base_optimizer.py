import logging
from pathlib import Path
from hpolib.abstract_benchmark import AbstractBenchmark

logger = logging.getLogger('Optimizer')


class Optimizer:
    def __init__(self, benchmark: AbstractBenchmark, optimizer_settings, benchmark_settings, intensifier, rng=0):
        self.benchmark = benchmark
        self.cs = benchmark.get_configuration_space()
        self.rng = rng
        self.optimizer_settings = optimizer_settings
        self.benchmark_settings = benchmark_settings
        self.intensifier = intensifier

    def setup(self):
        raise NotImplementedError()

    def run(self) -> Path:
        raise NotImplementedError()
