import logging
from typing import Union, Optional

import numpy as np
from hpbandster.core.worker import Worker

logger = logging.getLogger('Optimizer Utils')


class CustomWorker(Worker):
    def __init__(self, benchmark, benchmark_settings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark_settings = benchmark_settings

    def compute(self, config, budget, **kwargs):
        fidelity = {self.benchmark_settings['fidelity_name']: self.benchmark_settings['fidelity_type'](budget)}

        result_dict = self.benchmark.objective_function(config, **fidelity, **self.benchmark_settings)
        return {'loss': result_dict['function_value'],
                # TODO: add result dict in a generic fashion with also "non-pickable" return types.
                'info': {k: v for k, v in result_dict.items()}
                }


def get_sh_ta_runs(min_budget: Union[int, float], max_budget: Union[int, float], eta: int, n0: Optional[int] = None) \
        -> int:
    """ returns total no. of configurations for a given SH configuration """
    sh_iters = int(np.round((np.log(max_budget) - np.log(min_budget)) / np.log(eta), 8))
    if not n0:
        n0 = int(eta ** sh_iters)
    n_configs = n0 * np.power(eta, -np.linspace(0, sh_iters, sh_iters + 1))
    return int(sum(n_configs))


def get_number_ta_runs(iterations: int, min_budget: Union[int, float], max_budget: Union[int, float], eta: int) -> int:
    """ returns total no. of configurations (ta runs) for a given HB configuration """
    s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))

    all_s = list(range(s_max+1))[::-1]
    hb_iters = [all_s[i % (s_max + 1)] for i in range(iterations)]

    ta_runs = 0
    for s in hb_iters:
        n0 = int(np.floor((s_max + 1) / (s + 1)) * eta ** s)
        hb_min = eta ** -s * max_budget
        ta_runs += get_sh_ta_runs(hb_min, max_budget, eta, n0)
    return int(ta_runs)
