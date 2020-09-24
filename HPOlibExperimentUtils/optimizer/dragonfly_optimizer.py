import logging
from pathlib import Path
from typing import Union, Dict

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOlibExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer

logger = logging.getLogger('Optimizer')


class DragonflyOptimizer(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

    def setup(self):
        pass

    def run(self) -> None:
        """
        TODO: DRAGONFLY - This is the skeleton for the dragonfly optimizer.

        Returns
        -------
        None
        """
        pass