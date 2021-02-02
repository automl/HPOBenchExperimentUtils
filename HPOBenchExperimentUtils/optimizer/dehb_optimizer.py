"""
How to use the DEHB Optimizer

1) Download the Source Code
git clone https://github.com/automl/DEHB.git

 #TODO: Update to new version.
# We are currently using the first version of it.
cd DEHB
git checkout b8dcba7b38bf6e7fc8ce3e84ea567b66132e0eb5

2) Add the project to your Python Path
export PYTHONPATH=~/DEHB:$PYTHONPATH

3) Requirements
- dask distributed:
```
conda install dask distributed -c conda-forge
```
OR
```
python -m pip install dask distributed --upgrade
```

- Other things to install:
```
pip install numpy, ConfigSpace
```

"""

import logging
from pathlib import Path
from typing import Union, Dict
import sys
import numpy as np
import json
import ConfigSpace as CS


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

        #assert isinstance(self.main_fidelity, UniformFloatHyperparameter), \
        #    "DEHB only supports UniformFloat hyperparameters as main fidelity, received %s " % self.main_fidelity

        # Common objective function for DE & DEHB representing the benchmark
        def f(config: Configuration, budget=None):
            nonlocal self
            if budget is not None:
                if isinstance(self.main_fidelity, CS.hyperparameters.UniformIntegerHyperparameter) \
                        or isinstance(self.main_fidelity, CS.hyperparameters.NormalIntegerHyperparameter) \
                        or isinstance(self.main_fidelity.default_value, int):
                    budget = int(budget)
                else:
                    budget = float(budget)
            
                res = benchmark.objective_function(config,
                                                   fidelity={self.main_fidelity.name: budget},
                                                   **self.settings_for_sending,
                                                   )
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
