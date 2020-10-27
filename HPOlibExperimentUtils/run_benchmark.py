import logging
from multiprocessing import Process, Manager, Value
from pathlib import Path
from time import time, sleep
from typing import Union, Dict

from hpolib.util.example_utils import set_env_variables_to_use_only_one_core

try:
    from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
except:
    import sys, os.path
    sys.path.append(os.path.expandvars('$HPOEXPUTIL_PATH'))
    from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper

from HPOlibExperimentUtils.utils import PING_OPTIMIZER_IN_S
from HPOlibExperimentUtils.utils.optimizer_utils import get_optimizer, optimizer_str_to_enum, OptimizerEnum
from HPOlibExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
    load_benchmark, get_benchmark_names, get_optimizer_settings_names, \
    get_optimizer_setting

from HPOlibExperimentUtils import _log as _main_log, _default_log_format
_main_log.setLevel(logging.INFO)
# from HPOlibExperimentUtils.utils import _log as _utils_log
# _utils_log.setLevel(logging.DEBUG)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)

set_env_variables_to_use_only_one_core()
# enable_container_debug()


def run_benchmark(optimizer: Union[OptimizerEnum, str],
                  benchmark: str,
                  output_dir: Union[Path, str],
                  rng: int,
                  use_local: Union[bool, None] = False,
                  debug: bool = False,
                  **benchmark_params: Dict):
    """
    Run a HPOlib3 benchmark on a given Optimizer. Currently only SMAC, BOHB and Dragonfly are available as Optimizer.
    The benchmarks are by default stored in singularity container which are downloaded at the first run.

    Parameters
    ----------
    optimizer : str
        A string describing an optimizer with an setting. Those are defined in the optimizer_settings.yaml file
    benchmark : str
        This benchmark is selected from the HPOlib3 and executed. with the optimizer from above.
        Please have a look at the experiment_settings.json file to see what benchmarks are available.
        Incomplete list of supported benchmarks:
        'cartpolereduced', 'cartpolefull',
        'xgboost',
        'learna', 'metalearna',
        'NASCifar10ABenchmark', 'NASCifar10BBenchmark', 'NASCifar10CBenchmark',
        'SliceLocalizationBenchmark', 'ProteinStructureBenchmark',
            'NavalPropulsionBenchmark', 'ParkinsonsTelemonitoringBenchmark'
    output_dir : str, Path
        Directory where the optimizer stores its results. In this directory a result directory will be created
        with the format <optimizer_name>_run_<rng>.
    rng : int, None
        Random seed for the experiment. Also changes the output directory. By default 0.
    use_local : bool, None
        If you want to use the HPOlib3 benchamrks in a non-containerizd version (installed locally inside the
        current python environment), you can set this parameter to True. This is not recommend.
    benchmark_params : Dict
        Some benchmarks take special parameters for the initialization. For example, The XGBOostBenchmark takes as
        input a task_id. This task_id specifies the OpenML dataset to use.

        Please take a look into the HPOlib3 Benchamarks to find out if the benchmark needs further parameter.
        Note: Most of them dont need further parameter.
    """

    _log.info(f'Start running benchmark {benchmark} with optimizer setting {optimizer}.')

    if debug:
        _main_log.setLevel(level=logging.DEBUG)

    optimizer_settings = get_optimizer_setting(optimizer)
    benchmark_settings = get_benchmark_settings(benchmark)
    settings = dict(optimizer_settings, **benchmark_settings)
    _log.debug(f'Settings loaded: {settings}')

    optimizer_enum = optimizer_str_to_enum(optimizer_settings['optimizer'])
    _log.debug(f'Optimizer: {optimizer_enum}')

    output_dir = Path(output_dir) / benchmark / optimizer / f'run-{rng}'
    output_dir.mkdir(exist_ok=True, parents=True)
    _log.debug(f'Output dir: {output_dir}')

    # Load and instantiate the benchmark
    benchmark_obj = load_benchmark(benchmark_name=settings['import_benchmark'],
                                   import_from=settings['import_from'],
                                   use_local=use_local)

    if use_local:
        benchmark = benchmark_obj(rng=rng, **benchmark_params)
    else:
        from hpolib import config_file
        container_source = config_file.container_source
        benchmark = benchmark_obj(rng=rng, container_source=container_source, **benchmark_params)
    _log.info(f'Benchmark successfully initialized.')

    # Create a Process Manager to get access to the variable "total_time_proxy" of the bookkeeper
    # This variable represents how much time the optimizer has used
    manager = Manager()
    total_time_proxy = manager.Value('f', 0)
    # total_time_proxy = Value('f', 0)

    benchmark = Bookkeeper(benchmark,
                           output_dir,
                           total_time_proxy,
                           wall_clock_limit_in_s=settings['time_limit_in_s'],
                           cutoff_limit_in_s=settings['cutoff_in_s'],
                           is_surrogate=settings['is_surrogate'])

    _log.info(f'BookKeeper initialized. Additional benchmark parameters {benchmark_params}')

    optimizer = get_optimizer(optimizer_enum)
    optimizer = optimizer(benchmark=benchmark,
                          settings=settings,
                          output_dir=output_dir,
                          rng=rng)
    _log.info(f'Optimizer initialized. Start optimization process.')

    start_time = time()
    # Currently no optimizer uses the setup function, but we still call it to enable future optimizer implementations to
    # have a setup function.
    optimizer.setup()

    process = Process(target=optimizer.run, args=(), kwargs=dict())
    process.start()

    # Wait for the optimizer to finish. But in case the optimizer crashes somehow, also test for the real time here.
    while settings['time_limit_in_s'] >= benchmark.get_total_time_used() \
            and settings['time_limit_in_s'] >= (time() - start_time - 60) \
            and process.is_alive():
        sleep(PING_OPTIMIZER_IN_S)
    else:
        process.terminate()
        _log.info(f'Timelimit: {settings["time_limit_in_s"]} and is now: {benchmark.get_total_time_used()}')
        _log.info(f'Terminate Process after {time() - start_time}')

    _log.info(f'Run Benchmark - Finished.')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 running different benchmarks on different optimizer with a '
                                                 'unified interface')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--optimizer', choices=get_optimizer_settings_names(),  required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)
    parser.add_argument('--use_local', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False, help="When given, enables debug mode logging.")
    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    run_benchmark(**vars(args), **benchmark_params)
