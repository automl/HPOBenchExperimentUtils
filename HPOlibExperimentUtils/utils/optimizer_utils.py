import json
import logging
from enum import Enum
from typing import Union, Optional, Any, Dict

import numpy as np

from HPOlibExperimentUtils.utils.runner_utils import get_optimizer_settings_names

logger = logging.getLogger('Optimizer Utils')


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


class OptimizerEnum(Enum):
    """ Enumeration type for the supported optimizers """
    def __str__(self):
        return str(self.value)

    HPBANDSTER_HB = 'hpbandster_hyperband'
    HPBANDSTER_H2BO = 'hpbandster_h2bo'
    HPBANDSTER_RS = 'hpbandster_randomsearch'
    HPBANDSTER_BOHB = 'hpbandster_bohb'
    SMAC_HYPERBAND = 'smac_hyperband'
    SMAC_SUCCESSIVE_HALVING = 'smac_successive_halving'
    DRAGONFLY = 'dragonfly'


def optimizer_str_to_enum(optimizer: Union[OptimizerEnum, str]) -> OptimizerEnum:
    """
    Maps a name as string or enumeration typ of an optimizer to the enumeration object.

    Parameters
    ----------
    optimizer : Union[OptimizerEnum, str]
        If the type is 'str': return the optimizer-enumeration object.
        But if it is already the optimizer enumeration, just return the type again.

    Returns
    -------
        OptimizerEnum
    """
    if isinstance(optimizer, OptimizerEnum):
        return optimizer

    fail = False
    if isinstance(optimizer, str):
        if 'hpbandster' in optimizer:
            if 'bohb' in optimizer:
                return OptimizerEnum.HPBANDSTER_BOHB
            elif 'hyperband' in optimizer or 'hb' in optimizer:
                return OptimizerEnum.HPBANDSTER_HB
            elif 'randomsearch' in optimizer or 'rs' in optimizer:
                return OptimizerEnum.HPBANDSTER_RS
            elif 'h2bo' in optimizer:
                return OptimizerEnum.HPBANDSTER_H2BO
            else:
                fail = True

        elif 'smac' in optimizer:
            if 'hyperband' in optimizer or 'hb' in optimizer:
                return OptimizerEnum.SMAC_HYPERBAND
            elif 'successive_halving' in optimizer or 'sh' in optimizer:
                return OptimizerEnum.SMAC_SUCCESSIVE_HALVING
            else:
                fail = True

        elif 'dragonfly' in optimizer or 'df' == optimizer:
            return OptimizerEnum.DRAGONFLY

        else:
            fail = True
    else:
        raise TypeError(f'Unknown optimizer type. Must be one of str|OptimizerEnum, but was {type(optimizer)}')

    if fail:
        raise ValueError(f'Unknown optimizer str. Must be one of {get_optimizer_settings_names()},'
                         f' but was {optimizer}')


def get_optimizer(optimizer_enum):

    if optimizer_enum is OptimizerEnum.HPBANDSTER_BOHB:
        from HPOlibExperimentUtils.optimizer.bohb_optimizer import HpBandSterBOHBOptimizer
        optimizer = HpBandSterBOHBOptimizer
    elif optimizer_enum is OptimizerEnum.HPBANDSTER_RS:
        from HPOlibExperimentUtils.optimizer.bohb_optimizer import HpBandSterRandomSearchOptimizer
        optimizer = HpBandSterRandomSearchOptimizer
    elif optimizer_enum is OptimizerEnum.HPBANDSTER_HB:
        from HPOlibExperimentUtils.optimizer.bohb_optimizer import HpBandSterHyperBandOptimizer
        optimizer = HpBandSterHyperBandOptimizer
    elif optimizer_enum is OptimizerEnum.HPBANDSTER_H2BO:
        from HPOlibExperimentUtils.optimizer.bohb_optimizer import HpBandSterH2BOOptimizer
        optimizer = HpBandSterH2BOOptimizer

    elif optimizer_enum is OptimizerEnum.DRAGONFLY:
        from HPOlibExperimentUtils.optimizer.dragonfly_optimizer import DragonflyOptimizer
        optimizer = DragonflyOptimizer

    elif optimizer_enum is OptimizerEnum.SMAC_HYPERBAND:
        from HPOlibExperimentUtils.optimizer.smac_optimizer import SMACOptimizerHyperband
        optimizer = SMACOptimizerHyperband
    elif optimizer_enum is OptimizerEnum.SMAC_SUCCESSIVE_HALVING:
        from HPOlibExperimentUtils.optimizer.smac_optimizer import SMACOptimizerSuccessiveHalving
        optimizer = SMACOptimizerSuccessiveHalving

    else:
        raise ValueError(f'Unknown optimizer: {optimizer_enum}')
    return optimizer


def get_main_fidelity(fidelity_space, settings):
    """Helper function to get the main fidelity from a fidelity space. """
    if len(fidelity_space.get_hyperparameters()) > 1 and 'main_fidelity' not in settings:
        raise ValueError('Ok something went wrong. Please specify a main fidelity in the benchmark settings')

    if 'main_fidelity' in settings:
        main_fidelity = settings['main_fidelity']
        fidelity = fidelity_space.get_hyperparameter(main_fidelity)
    else:
        fidelity = fidelity_space.get_hyperparameters()[0]
    return fidelity