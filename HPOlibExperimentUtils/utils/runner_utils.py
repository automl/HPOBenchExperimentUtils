import json
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Dict, Tuple

logger = logging.getLogger('Runner Utils')


class OptimizerEnum(Enum):
    """ Enumeration type for the supported optimizers """
    def __str__(self):
        return str(self.value)

    BOHB = 'bohb'
    HYPERBAND = 'hyperband'
    SUCCESSIVE_HALVING = 'successive_halving'
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
    if isinstance(optimizer, str):
        if 'BOHB' in optimizer.upper():
            return OptimizerEnum.BOHB
        elif 'HYPERBAND' in optimizer.upper() or 'HB' == optimizer.upper():
            return OptimizerEnum.HYPERBAND
        elif 'SUCCESSIVE_HALVING' in optimizer.upper() or 'SH' == optimizer.upper():
            return OptimizerEnum.SUCCESSIVE_HALVING
        elif 'DRAGONFLY' in optimizer.upper() or 'DF' == optimizer.upper():
            return OptimizerEnum.DRAGONFLY
        else:
            raise ValueError(f'Unknown optimizer str. Must be one of BOHB|SMAC|DRAGONFLY, but was {optimizer}')
    else:
        raise TypeError(f'Unknown optimizer type. Must be one of str|OptimizerEnum, but was {type(optimizer)}')


def transform_unknown_params_to_dict(unknown_args: List) -> Dict:
    """
    Given a list of unknown parameters in form ['--name', '--value', ...], it transforms the list into a dictionary.
    TODO (pm): figure out how to find the right parameter type. Currently, it just casts it to an integer.

    This function is used to extract the benchmark parameters (such as the task id for the xgboost benchmark) from the
    command line arguments.

    Parameters
    ----------
    unknown_args : List

    Returns
    -------
    Dict
    """
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


def get_setting_per_benchmark(benchmark: str, rng: int, output_dir: Path) -> Tuple[Dict, Dict]:
    """
    Get a dictionary for each benchmark containing the experiment parameters.

    Parameters
    ----------
    benchmark : str
    rng : int
        Random seed
    output_dir : Path
        path to the output directory

    Returns
    -------
        Tuple[Dict, Dict] - optimizer settings, benchmark settings
    """
    experiment_settings_path = Path(__file__).absolute().parent.parent / 'experiment_settings.json'

    with experiment_settings_path.open('r') as fh:
        experiment_settings = json.load(fh)

    assert benchmark.lower() in experiment_settings.keys(),\
        f"benchmark name {benchmark.lower()} not found. Should be one of {', '.join(experiment_settings.keys())}"

    # TODO: DRAGONFLY - The experiment_settings.json contain also settings for the optimizer to make them comparable,
    #       s.a. min or maximum budget and available runtime. It would be good to map the Dragonfly parameters to the
    #       same parameters, such that the optimizers still comparable.
    optimizer_settings = experiment_settings[benchmark.lower()]['optimizer_settings']
    benchmark_settings = experiment_settings[benchmark.lower()]['benchmark_settings']

    optimizer_settings.update({'rng': rng, 'output_dir': output_dir})
    benchmark_settings.update({'rng': rng, 'output_dir': output_dir})

    return optimizer_settings, benchmark_settings
