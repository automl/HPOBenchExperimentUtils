import logging
from multiprocessing import Process, Manager, Value
from pathlib import Path
from time import time, sleep
from typing import Union, Dict

try:
    from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
except:
    import sys, os.path
    sys.path.append(os.path.expandvars('$HPOEXPUTIL_PATH'))

from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper, total_time_exceeds_limit, used_fuel_exceeds_limit, \
    tae_exceeds_limit
from HPOBenchExperimentUtils.utils import PING_OPTIMIZER_IN_S
from HPOBenchExperimentUtils.utils.optimizer_utils import get_optimizer, optimizer_str_to_enum
from HPOBenchExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
    load_benchmark, get_benchmark_names, get_optimizer_settings_names, \
    get_optimizer_setting
from HPOBenchExperimentUtils.extract_trajectory import extract_trajectory

from HPOBenchExperimentUtils import _log as _main_log, _default_log_format

_main_log.setLevel(logging.INFO)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


def run_benchmark(optimizer: str,
                  benchmark: str,
                  output_dir: Union[Path, str],
                  rng: int,
                  use_local: Union[bool, None] = False,
                  debug: bool = False,
                  **benchmark_params: Dict):
    """
    Run a HPOBench benchmark on a given Optimizer. Currently only SMAC, BOHB and Dragonfly are available as Optimizer.
    The benchmarks are by default stored in singularity container which are downloaded at the first run.

    Parameters
    ----------
    optimizer : str
        A string describing an optimizer with an setting. Those are defined in the optimizer_settings.yaml file
    benchmark : str
        This benchmark is selected from the HPOBench and executed. with the optimizer from above.
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
        If you want to use the HPOBench benchamrks in a non-containerizd version (installed locally inside the
        current python environment), you can set this parameter to True. This is not recommend.
    benchmark_params : Dict
        Some benchmarks take special parameters for the initialization. For example, The XGBOostBenchmark takes as
        input a task_id. This task_id specifies the OpenML dataset to use.

        Please take a look into the HPOBench Benchamarks to find out if the benchmark needs further parameter.
        Note: Most of them dont need further parameter.
    """

    _log.info(f'Start running benchmark {benchmark} with optimizer setting {optimizer}.')

    if debug:
        _main_log.setLevel(level=logging.DEBUG)
        from hpobench.util.container_utils import enable_container_debug
        enable_container_debug()

    optimizer_settings = get_optimizer_setting(optimizer)
    benchmark_settings = get_benchmark_settings(benchmark)
    settings = dict(optimizer_settings, **benchmark_settings)
    _log.debug(f'Settings loaded: {settings}')

    optimizer_enum = optimizer_str_to_enum(optimizer_settings['optimizer'])
    _log.debug(f'Optimizer: {optimizer_enum}')

    if "task_id" in benchmark_params:
        dname = "%s/%d" % (benchmark, benchmark_params["task_id"])
    else:
        dname = benchmark

    output_dir = Path(output_dir) / dname / optimizer / f'run-{rng}'
    output_dir = output_dir.absolute()
    if output_dir.is_dir():
        raise ValueError("Outputdir %s already exists, pass" % output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)  # TODO!

    _log.debug(f'Output dir: {output_dir}')

    # Load and instantiate the benchmark
    benchmark_obj = load_benchmark(benchmark_name=settings['import_benchmark'],
                                   import_from=settings['import_from'],
                                   use_local=use_local)

    if use_local:
        benchmark = benchmark_obj(rng=rng, **benchmark_params)
    else:
        from hpobench import config_file
        container_source = config_file.container_source
        benchmark = benchmark_obj(rng=rng, container_source=container_source, **benchmark_params)
    _log.info(f'Benchmark successfully initialized.')

    # Create a Process Manager to get access to the proxy variables. They represent the state of the optimization
    # process.
    manager = Manager()

    global_lock = manager.Lock()
    # from multiprocessing import Lock
    # global_lock = Lock()

    # This variable count the time the benchmark was evaluated (in seconds).
    # This is the cost of the objective function + the overhead. Use type double.
    total_time_proxy = manager.Value('d', 0)
    # total_time_proxy = Value('d', 0)

    # We can also restrict how often a optimizer is allowed to execute the target algorithm. Encode it as type long.
    total_tae_calls_proxy = manager.Value('l', 0)
    # total_tae_calls_proxy = Value('l', 0)

    # Or we can give an upper limit for amount of budget, the optimizer can use. One can think of it as fuel. Running a
    # benchmark reduces the fuel by `budget`. Use type double.
    total_fuel_used_proxy = manager.Value('d', 0)
    # total_fuel_used_proxy = Value('d', 0)

    benchmark = Bookkeeper(benchmark=benchmark,
                           output_dir=output_dir,
                           total_time_proxy=total_time_proxy,
                           total_tae_calls_proxy=total_tae_calls_proxy,
                           total_fuel_used_proxy=total_fuel_used_proxy,
                           global_lock=global_lock,
                           wall_clock_limit_in_s=settings['time_limit_in_s'],
                           tae_limit=settings['tae_limit'],
                           fuel_limit=settings['fuel_limit'],
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

    while not total_time_exceeds_limit(benchmark.get_total_time_used(), settings['time_limit_in_s'], start_time) \
            and not tae_exceeds_limit(benchmark.get_total_tae_used(), settings['tae_limit']) \
            and not used_fuel_exceeds_limit(benchmark.get_total_fuel_used(), settings['fuel_limit']) \
            and process.is_alive():
        sleep(PING_OPTIMIZER_IN_S)
    else:
        process.terminate()
        _log.info(f'Optimization has been finished.\n'
                  f'Timelimit: {settings["time_limit_in_s"]} and is now: {benchmark.get_total_time_used()}\n'
                  f'TAE limit: {settings["tae_limit"]} and is now: {benchmark.get_total_tae_used()}\n'
                  f'Fuel limit: {settings["fuel_limit"]} and is now: {benchmark.get_total_fuel_used()}\n'
                  f'Terminate Process after {time() - start_time}')

    optimizer.shutdown()

    _log.info(f'Extract the trajectories')
    extract_trajectory(output_dir=output_dir, debug=debug)

    _log.info(f'Run Benchmark - Finished.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='HPOBench Wrapper',
                                     description='HPOBench running different benchmarks on different optimizer with a '
                                                 'unified interface')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--optimizer', choices=get_optimizer_settings_names(), required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)
    parser.add_argument('--use_local', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False, help="When given, enables debug mode logging.")
    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    run_benchmark(**vars(args), **benchmark_params)
