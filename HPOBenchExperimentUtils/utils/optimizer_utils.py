import json
import logging
from enum import Enum
from typing import Union, Optional, Any, Dict

import numpy as np

from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_settings_names

_log = logging.getLogger(__name__)


class OptimizerEnum(Enum):
    """ Enumeration type for the supported optimizers """
    def __str__(self):
        return str(self.value)

    HPBANDSTER_HB = 'hpbandster_hyperband'
    HPBANDSTER_BOHB = 'hpbandster_bohb'
    HPBANDSTER_TPE = 'hpbandster_tpe'
    SMAC_SF = "smac_sf"
    SMAC_BO = "smac_bo"
    SMAC_HYPERBAND = 'smac_hb'
    SMAC_SUCCESSIVE_HALVING = 'smac_sh'
    DRAGONFLY = 'dragonfly'
    DEHB = 'dehb'
    DE = 'de'
    FABOLAS = 'fabolas'
    MUMBO = 'mumbo'
    PURE_RANDOMSEARCH = 'randomsearch'
    AUTOGLUON = 'autogluon'
    RAY_HYPEROPT_ASHA = 'ray_hyperopt_asha'
    RAY_BAYESOPT_ASHA = 'ray_bayesopt_asha'
    RAY_HYPEROPT_NO_FIDELITY = 'ray_hyperopt'
    RAY_RANDOMSEARCH = 'ray_randomsearch'
    OPTUNA_TPE_ASHA = 'optuna_tpe_asha'
    OPTUNA_CMAES_ASHA = 'optuna_cmaes_asha'
    OPTUNA_RANDOMSEARCH = 'optuna_randomsearch'
    OPTUNA_TPE_MEDIAN = 'optuna_tpe_median'


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
            elif 'hb' in optimizer:
                return OptimizerEnum.HPBANDSTER_HB
            elif 'tpe' in optimizer:
                return OptimizerEnum.HPBANDSTER_TPE
            else:
                fail = True

        elif 'smac' in optimizer:
            if 'hb' in optimizer:
                return OptimizerEnum.SMAC_HYPERBAND
            elif 'sh' in optimizer:
                return OptimizerEnum.SMAC_SUCCESSIVE_HALVING
            elif 'sf' in optimizer:
                return OptimizerEnum.SMAC_SF
            elif 'bo' in optimizer:
                return OptimizerEnum.SMAC_BO
            else:
                fail = True

        elif 'dragonfly' in optimizer:
            return OptimizerEnum.DRAGONFLY

        elif optimizer == 'dehb':
            return OptimizerEnum.DEHB

        elif optimizer == 'de':
            return OptimizerEnum.DE

        elif 'fabolas' in optimizer:
            return OptimizerEnum.FABOLAS

        elif optimizer == 'mumbo':
            return OptimizerEnum.MUMBO

        elif optimizer == "autogluon":
            return OptimizerEnum.AUTOGLUON

        elif optimizer == 'randomsearch':
            return OptimizerEnum.PURE_RANDOMSEARCH

        elif optimizer == 'ray_hyperopt_asha':
            return OptimizerEnum.RAY_HYPEROPT_ASHA
        elif optimizer == 'ray_bayesopt_asha':
            return OptimizerEnum.RAY_BAYESOPT_ASHA
        elif optimizer == 'ray_hyperopt':
            return OptimizerEnum.RAY_HYPEROPT_NO_FIDELITY
        elif optimizer == 'ray_randomsearch':
            return OptimizerEnum.RAY_RANDOMSEARCH

        elif optimizer == 'optuna_tpe_asha':
            return OptimizerEnum.OPTUNA_TPE_ASHA
        elif optimizer == 'optuna_cmaes_asha':
            return OptimizerEnum.OPTUNA_CMAES_ASHA
        elif optimizer == 'optuna_randomsearch':
            return OptimizerEnum.OPTUNA_RANDOMSEARCH
        elif optimizer == 'optuna_tpe_median':
            return OptimizerEnum.OPTUNA_TPE_MEDIAN

        else:
            fail = True
    else:
        raise TypeError(f'Unknown optimizer type. Must be one of str|OptimizerEnum, but was {type(optimizer)}')

    if fail:
        raise ValueError(f'Unknown optimizer str. Must be one of {get_optimizer_settings_names()},'
                         f' but was {optimizer}')


def get_optimizer(optimizer_enum):

    if optimizer_enum is OptimizerEnum.HPBANDSTER_BOHB:
        from HPOBenchExperimentUtils.optimizer.bohb_optimizer import HpBandSterBOHBOptimizer
        optimizer = HpBandSterBOHBOptimizer
    elif optimizer_enum is OptimizerEnum.HPBANDSTER_HB:
        from HPOBenchExperimentUtils.optimizer.bohb_optimizer import HpBandSterHyperBandOptimizer
        optimizer = HpBandSterHyperBandOptimizer
    elif optimizer_enum is OptimizerEnum.HPBANDSTER_TPE:
        from HPOBenchExperimentUtils.optimizer.bohb_optimizer import HpBandSterTPEOptimizer
        optimizer = HpBandSterTPEOptimizer
    elif optimizer_enum is OptimizerEnum.DRAGONFLY:
        from HPOBenchExperimentUtils.optimizer.dragonfly_optimizer import DragonflyOptimizer
        optimizer = DragonflyOptimizer
    elif optimizer_enum is OptimizerEnum.DEHB:
        from HPOBenchExperimentUtils.optimizer.dehb_optimizer import DehbOptimizer
        optimizer = DehbOptimizer
    elif optimizer_enum is OptimizerEnum.DE:
        from HPOBenchExperimentUtils.optimizer.dehb_optimizer import DeOptimizer
        optimizer = DeOptimizer
    elif optimizer_enum is OptimizerEnum.FABOLAS:
        from HPOBenchExperimentUtils.optimizer.fabolas_optimizer import FabolasOptimizer
        optimizer = FabolasOptimizer
    elif optimizer_enum is OptimizerEnum.MUMBO:
        from HPOBenchExperimentUtils.optimizer.mumbo import MultiTaskMUMBO
        optimizer = MultiTaskMUMBO
    elif optimizer_enum is OptimizerEnum.SMAC_HYPERBAND:
        from HPOBenchExperimentUtils.optimizer.smac_optimizer import SMACOptimizerHyperband
        optimizer = SMACOptimizerHyperband
    elif optimizer_enum is OptimizerEnum.SMAC_SF:
        from HPOBenchExperimentUtils.optimizer.smac_optimizer import SMACOptimizerHPO
        optimizer = SMACOptimizerHPO
    elif optimizer_enum is OptimizerEnum.SMAC_BO:
        from HPOBenchExperimentUtils.optimizer.smac_optimizer import SMACOptimizerBO
        optimizer = SMACOptimizerBO
    elif optimizer_enum is OptimizerEnum.SMAC_SUCCESSIVE_HALVING:
        from HPOBenchExperimentUtils.optimizer.smac_optimizer import SMACOptimizerSuccessiveHalving
        optimizer = SMACOptimizerSuccessiveHalving
    elif optimizer_enum is OptimizerEnum.AUTOGLUON:
        from HPOBenchExperimentUtils.optimizer.autogluon_optimizer import AutogluonOptimizer
        optimizer = AutogluonOptimizer
    elif optimizer_enum is OptimizerEnum.PURE_RANDOMSEARCH:
        from HPOBenchExperimentUtils.optimizer.randomsearch_optimizer import RandomSearchOptimizer
        optimizer = RandomSearchOptimizer
    elif optimizer_enum is OptimizerEnum.RAY_HYPEROPT_ASHA:
        from HPOBenchExperimentUtils.optimizer.ray_optimizer import RayHBHyperoptOptimizer
        optimizer = RayHBHyperoptOptimizer
    elif optimizer_enum is OptimizerEnum.RAY_BAYESOPT_ASHA:
        from HPOBenchExperimentUtils.optimizer.ray_optimizer import RayHBBayesOptOptimizer
        optimizer = RayHBBayesOptOptimizer
    elif optimizer_enum is OptimizerEnum.RAY_HYPEROPT_NO_FIDELITY:
        from HPOBenchExperimentUtils.optimizer.ray_optimizer import RayHyperoptWithoutFidelityOptimizer
        optimizer = RayHyperoptWithoutFidelityOptimizer
    elif optimizer_enum is OptimizerEnum.RAY_RANDOMSEARCH:
        from HPOBenchExperimentUtils.optimizer.ray_optimizer import RayRandomSearchOptimizer
        optimizer = RayRandomSearchOptimizer
    elif optimizer_enum is OptimizerEnum.OPTUNA_TPE_ASHA:
        from HPOBenchExperimentUtils.optimizer.optuna_optimizer import OptunaTPEHyperbandOptimizer
        optimizer = OptunaTPEHyperbandOptimizer
    elif optimizer_enum is OptimizerEnum.OPTUNA_CMAES_ASHA:
        from HPOBenchExperimentUtils.optimizer.optuna_optimizer import OptunaCMAESHyperBandOptimizer
        optimizer = OptunaCMAESHyperBandOptimizer
    elif optimizer_enum is OptimizerEnum.OPTUNA_RANDOMSEARCH:
        from HPOBenchExperimentUtils.optimizer.optuna_optimizer import OptunaRandomSearchOptimizer
        optimizer = OptunaRandomSearchOptimizer
    elif optimizer_enum is OptimizerEnum.OPTUNA_TPE_MEDIAN:
        from HPOBenchExperimentUtils.optimizer.optuna_optimizer import OptunaTPEMedianStoppingOptimizer
        optimizer = OptunaTPEMedianStoppingOptimizer
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_enum}')
    return optimizer


def get_main_fidelity(fidelity_space, settings):
    """Helper function to get the main fidelity from a fidelity space. """
    if len(fidelity_space.get_hyperparameters()) > 1 and 'main_fidelity' not in settings:
        raise ValueError('Something went wrong. Please specify a main fidelity in the benchmark settings')

    if 'main_fidelity' in settings:
        main_fidelity = settings['main_fidelity']
        fidelity = fidelity_space.get_hyperparameter(main_fidelity)
    else:
        fidelity = fidelity_space.get_hyperparameters()[0]
    return fidelity


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
            _log.warning(f'Value of {key} is not json-serializable. Type was: {type(value)}')
        else:
            benchmark_dict_for_sending[key] = value
    return benchmark_dict_for_sending