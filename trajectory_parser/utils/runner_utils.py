import logging
from enum import Enum
from pathlib import Path
from typing import List, Union

logger = logging.getLogger('Runner Utils')


class OptimizerEnum(Enum):

    def __str__(self):
        return str(self.value)

    BOHB = 'bohb'
    SMAC = 'smac'
    HYPERBAND = 'hyperband'
    SUCCESSIVE_HALVING = 'succesive_halving'


def optimizer_str_to_enum(optimizer: Union[OptimizerEnum, str]):
    if isinstance(optimizer, OptimizerEnum):
        return optimizer
    if isinstance(optimizer, str):
        if 'BOHB' in optimizer.upper():
            return OptimizerEnum.BOHB
        elif 'SMAC' in optimizer.upper():
            return OptimizerEnum.SMAC
        elif 'HYPERBAND' in optimizer.upper() or 'HB' == optimizer.upper():
            return OptimizerEnum.HYPERBAND
        elif 'SUCCESSIVE_HALVING' in optimizer.upper() or 'SH' == optimizer.upper():
            return OptimizerEnum.SUCCESSIVE_HALVING
        else:
            raise ValueError(f'Unknown optimizer str. Must be one of BOHB|SMAC, but was {optimizer}')
    else:
        raise TypeError(f'Unknown optimizer type. Must be one of str|OptimizerEnum, but was {type(optimizer)}')


def transform_unknown_params_to_dict(unknown_args: List):
    benchmark_params = {}
    for i in range(0, len(unknown_args), 2):
        try:
            value = int(unknown_args[i+1])
        except ValueError:
            value = unknown_args[i+1]
        except IndexError:
            raise IndexError('While parsing additional arguments an index error occured. '
                             'This means a parameter has no value.')

        benchmark_params[unknown_args[i][2:]] = value
    return benchmark_params


def get_setting_per_benchmark(benchmark: str, seed: int, output_dir: Path):

    if 'cartpolereduced' in benchmark.lower() or 'cartpolefull' in benchmark.lower():
        optimizer_settings = {'min_budget': 1,
                              'max_budget': 9,
                              'num_iterations': 10,
                              'eta': 3,
                              'time_limit_in_s': 4000,
                              'cutoff_in_s': 1800,
                              'mem_limit_in_mb': 4000
                              }
        benchmark_settings = {'fidelity_name': 'budget',
                              'fidelity_type': int,
                              'import_from': 'rl.cartpole',
                              'import_benchmark': 'CartpoleReduced'
                                                  if 'cartpolereduced' in benchmark.lower() else 'CartpoleFull'
                              }

    elif 'xgboost' in benchmark.lower():
        optimizer_settings = {'min_budget': 0.1,
                              'max_budget': 1.0,
                              'eta': 3,
                              'num_iterations': 10,
                              'time_limit_in_s': 4000,
                              'cutoff_in_s': 1800,
                              'mem_limit_in_mb': 4000,
                              }

        benchmark_settings = {'fidelity_name': 'subsample',
                              'fidelity_type': float,
                              'n_estimators': 128,
                              'import_from': 'ml.xgboost_benchmark',
                              'import_benchmark': 'XGBoostBenchmark'
                              }
    else:
        raise ValueError(f'Unknown Benchmark {benchmark}')

    optimizer_settings.update({'seed': seed, 'output_dir': output_dir})
    benchmark_settings.update({'seed': seed, 'output_dir': output_dir})

    return optimizer_settings, benchmark_settings
