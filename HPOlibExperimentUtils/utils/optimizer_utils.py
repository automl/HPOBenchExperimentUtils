import json
import logging
from typing import Union, Optional, Any, Dict

import numpy as np

from HPOlibExperimentUtils.utils.runner_utils import OptimizerEnum

logger = logging.getLogger('Optimizer Utils')


def parse_fidelity_type(fidelity_type: str):
    """
    Helperfunction to cast the fidelity into its correct type. This step is necessary since we can only store the
    fidelity type in the experiment_settings.json as string.

    Parameters
    ----------
    fidelity_type : str
        The name of the type of the fidelity
    Returns
    -------
        the python object representing this type.
    """
    if fidelity_type.lower() == 'str':
        return str
    elif fidelity_type.lower() == 'int':
        return int
    elif fidelity_type.lower() == 'float':
        return float
    else:
        raise ValueError(f'Unknown fidelity type: {fidelity_type}. Must be one of [str, int, float].')


def get_sh_ta_runs(min_budget: Union[int, float], max_budget: Union[int, float], eta: int, n0: Optional[int] = None) \
        -> int:
    """ Returns total number of configurations for a given SH configuration """
    sh_iters = int(np.round((np.log(max_budget) - np.log(min_budget)) / np.log(eta), 8))
    if not n0:
        n0 = int(eta ** sh_iters)
    n_configs = n0 * np.power(eta, -np.linspace(0, sh_iters, sh_iters + 1))
    return int(sum(n_configs))


def get_number_ta_runs(iterations: int, min_budget: Union[int, float], max_budget: Union[int, float], eta: int) -> int:
    """ Returns the total number of configurations (ta runs) for a given HB configuration """
    s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))

    all_s = list(range(s_max+1))[::-1]
    hb_iters = [all_s[i % (s_max + 1)] for i in range(iterations)]

    ta_runs = 0
    for s in hb_iters:
        n0 = int(np.floor((s_max + 1) / (s + 1)) * eta ** s)
        hb_min = eta ** -s * max_budget
        ta_runs += get_sh_ta_runs(hb_min, max_budget, eta, n0)
    return int(ta_runs)


def is_jsonable(value: Any) -> bool:
    """ Helperfunction to check if a value is json serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def prepare_dict_for_sending(benchmark_settings: Dict):
    """
    Removes all non json-serializable parameter from a dictionary. A warning will be shown if a parameter is removed.
    Parameters
    ----------
    benchmark_settings : Dict
        Dict container parameters for running a benchmark. E.g. The seed and the output directory.
        However it may include things in a non-serializable data type.
    Returns
    -------
    Dict
    """
    benchmark_dict_for_sending = {}
    for key, value in benchmark_settings.items():
        if not is_jsonable(value):
            if key == 'output_dir':
                continue
            logger.warning(f'Value of {key} is not json-serializable. Type was: {type(value)}')
        else:
            benchmark_dict_for_sending[key] = value
    return benchmark_dict_for_sending


def get_optimizer(optimizer_enum):

    if optimizer_enum is OptimizerEnum.BOHB:
        from HPOlibExperimentUtils.optimizer.bohb_optimizer import BOHBOptimizer
        optimizer = BOHBOptimizer
    elif optimizer_enum is OptimizerEnum.DRAGONFLY:
        from HPOlibExperimentUtils.optimizer.dragonfly_optimizer import DragonflyOptimizer
        optimizer = DragonflyOptimizer
    elif optimizer_enum is OptimizerEnum.HYPERBAND:
        from HPOlibExperimentUtils.optimizer.smac_optimizer import SMACOptimizerHyperband
        optimizer = SMACOptimizerHyperband
    elif optimizer_enum is OptimizerEnum.SUCCESSIVE_HALVING:
        from HPOlibExperimentUtils.optimizer.smac_optimizer import SMACOptimizerSuccessiveHalving
        optimizer = SMACOptimizerSuccessiveHalving
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_enum}')
    return optimizer
