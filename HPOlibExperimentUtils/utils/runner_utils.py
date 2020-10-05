import logging
from importlib import import_module
from pathlib import Path
from typing import List, Dict, Any

import yaml

logger = logging.getLogger('Runner Utils')


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


def load_optimizer_settings() -> Dict:
    """ Load the experiment settings from file """
    optimizer_settings_path = Path(__file__).absolute().parent.parent / 'optimizer_settings.yaml'
    with optimizer_settings_path.open('r') as fh:
        optimizer_settings = yaml.load(fh, yaml.FullLoader)

    # Check that all mandatory fields are in the settings given:
    # mandatory = ['time_limit_in_s', 'cutoff_in_s', 'mem_limit_in_mb', 'import_from', 'import_benchmark']
    # assert all([option in optimizer_settings for option in mandatory])

    # fill missing parameter with default ones:
    default_params = dict(num_iterations=10000000,
                          is_surrogate=False)

    # Update the optimizer settings (this command is comparable to default_params.update(optimizer_settings)
    optimizer_settings = dict(default_params, **optimizer_settings)
    return optimizer_settings


def get_optimizer_settings_names():
    settings = load_optimizer_settings()
    return list(settings.keys())


def get_optimizer_setting(optimizer_setting_str: str) -> Dict:
    optimizer_settings = load_optimizer_settings()
    settings_names = get_optimizer_settings_names()

    assert optimizer_setting_str in settings_names,\
        f"Optimizer setting {optimizer_setting_str} not found. Should be one of {', '.join(settings_names)}"

    return optimizer_settings[optimizer_setting_str]


def load_benchmark_settings() -> Dict:
    """ Load the experiment settings from file """
    experiment_settings_path = Path(__file__).absolute().parent.parent / 'benchmark_settings.yaml'

    with experiment_settings_path.open('r') as fh:
        experiment_settings = yaml.load(fh, yaml.FullLoader)

    # Check that all mandatory fields are in the settings given:
    mandatory = ['time_limit_in_s', 'cutoff_in_s', 'mem_limit_in_mb', 'import_from', 'import_benchmark']
    for benchmark, settings in experiment_settings.items():
        found = [option in settings for option in mandatory]
        assert all(found), "Missing mandatory option(s) %s in benchmark settings %s" % \
                           (str([o for b, o in zip(found, mandatory) if not b]), str(settings))
    return experiment_settings


def get_benchmark_names():
    """ Get the names for the supported benchmarks. """
    experiment_settings = load_benchmark_settings()
    return list(experiment_settings.keys())


def get_benchmark_settings(benchmark: str) -> Dict:
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
    experiment_settings = load_benchmark_settings()
    benchmark_names = get_benchmark_names()

    assert benchmark in benchmark_names,\
        f"benchmark name {benchmark} not found. Should be one of {', '.join(benchmark_names)}"

    settings = experiment_settings[benchmark]
    settings['is_surrogate'] = settings.get('is_surrogate', False)

    return experiment_settings[benchmark]


def load_benchmark(benchmark_name, import_from, use_local: bool) -> Any:
    """
    Load the benchmark object.
    If not `use_local`:  Then load a container from a given source, defined in the Hpolib.

    Import via command from hpolib.[container.]benchmarks.<import_from> import <benchmark_name>

    Parameters
    ----------
    benchmark_name : str
    import_from : str
    use_local : bool
        By default this value is set to false.
        In this case, a container will be downloaded. This container includes all necessary files for the experiment.
        You don't have to install something.

        If true, use the experiment locally. Therefore the experiment has to be installed.
        See the experiment description in the HPOlib3.

    Returns
    -------
    Benchmark
    """
    import_str = 'hpolib.' + ('container.' if not use_local else '') + 'benchmarks.' + import_from
    logger.debug(f'Try to execute command: from {import_str} import {benchmark_name}')

    module = import_module(import_str)
    benchmark_obj = getattr(module, benchmark_name)
    logger.debug(f'Benchmark {benchmark_name} successfully loaded')

    return benchmark_obj
