import logging
from pathlib import Path
from typing import Dict, Any, Union, Type

import pandas as pd
import numpy as np

import ConfigSpace as CS

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer


_log = logging.getLogger(__name__)


class HEBOOptimizer(SingleFidelityOptimizer):
    """
    This class offers an interface to the HEBO Optimizer (https://github.com/huawei-noah/HEBO).

    Note: Trying to use the original python environment used for the BBO competition:
    https://github.com/huawei-noah/noah-research/blob/master/BO/HEBO/archived_submissions/hebo/install_order.txt
    https://github.com/rdturnermtl/bbo_challenge_starter_kit/blob/master/environment.txt

    Since:
    * higher version of pymoo crashes
    --> ModuleNotFoundError: No module named 'pymoo.algorithms.so_genetic_algorithm'
    * higher version of scipy crashes
    --> ValueError: Unsupported dtype object
    * higher version of torch crashes
    --> #ValueError: The value argument must be within the support
    """
    def __init__(self, benchmark: Bookkeeper, settings: Dict, output_dir: Path,
                 rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        self.hebo_space = self._transform_space(self.cs)

    def _transform_space(self, cs: CS.ConfigurationSpace):
        # Create hebo configspace
        hebo_dc = []
        for hp in cs.get_hyperparameters():
            param = None
            if isinstance(hp, CS.OrdinalHyperparameter):
                # HEBO does not handle ordinal hyperparameter, make them int
                param = {"name": hp.name, 'type': 'int', 'lb':0, 'ub': len(hp.sequence)-1}
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                if hp.log:
                    param = {"name": hp.name, 'type': 'pow_int', 'lb': hp.lower, 'ub': hp.upper}
                else:
                    param = {"name": hp.name, 'type': 'int', 'lb': hp.lower, 'ub': hp.upper}
            elif isinstance(hp, CS.UniformFloatHyperparameter):
                if hp.log:
                    param = {"name": hp.name, 'type': 'pow', 'lb': hp.lower, 'ub': hp.upper}
                else:
                    param = {"name": hp.name, 'type': 'int', 'lb': hp.lower, 'ub': hp.upper}
            elif isinstance(hp, CS.CategoricalHyperparameter):
                param = {"name": hp.name, 'type': 'cat', 'categories': hp.choices}
            else:
                raise NotImplementedError("Unknown Parameter Type", hp)
            hebo_dc.append(param)
        return hebo_dc

    def setup(self):
        pass

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """

        def optimization_function_wrapper(params : pd.DataFrame) -> np.ndarray:
            run_id = SingleFidelityOptimizer._id_generator()
            # Get dictionary
            params = params.iloc[0].to_dict()
            for h in self.cs.get_hyperparameters():
                if isinstance(h, CS.OrdinalHyperparameter):
                    params[h.name] = h.sequence[int(params[h.name])]

            result_dict = self.benchmark.objective_function(configuration=params,
                                                            configuration_id=run_id,
                                                            #fidelity=fidelity,
                                                            **self.settings_for_sending)
            res = np.array([result_dict['function_value'], ], dtype=float)
            print(res)
            return res

        print(self.hebo_space)
        print(self.cs)
        opt = HEBO(space=DesignSpace().parse(self.hebo_space))
        i = 1
        while True:
            rec = opt.suggest(n_suggestions=1)
            opt.observe(rec, optimization_function_wrapper(rec))
            print('After %d iterations, best obj is %g' % (i, opt.y.min()))
            i += 1
