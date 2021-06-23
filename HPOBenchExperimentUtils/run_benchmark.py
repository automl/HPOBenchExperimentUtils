import logging
from multiprocessing import Process, Value, Manager
from pathlib import Path
from time import time, sleep
from typing import Union, Dict

try:
    from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
except:
    import sys, os.path
    sys.path.append(os.path.expandvars('$HPOEXPUTIL_PATH'))

from HPOBenchExperimentUtils import _log as _root_log
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper, total_time_exceeds_limit, used_fuel_exceeds_limit, \
    tae_exceeds_limit
from HPOBenchExperimentUtils.extract_trajectory import extract_trajectory
from HPOBenchExperimentUtils.utils import PING_OPTIMIZER_IN_S
from HPOBenchExperimentUtils.utils.optimizer_utils import get_optimizer, optimizer_str_to_enum
from HPOBenchExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
    load_benchmark, get_benchmark_names, get_optimizer_settings_names, \
    get_optimizer_setting

_root_log.setLevel(logging.INFO)
_log = logging.getLogger(__name__)


def run_benchmark(optimizer: str,
                  benchmark: str,
                  output_dir: Union[Path, str],
                  rng: int,
                  resource_file_dir: Union[Path, str, None] = None,
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
    resource_file_dir : str, Path, None
        We keep track of the used resources by writing them into a resource file. This parameter specifies where to
        store this resource file. By default, this directory is set to the TEMP directory.
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
        _root_log.setLevel(level=logging.DEBUG)
        _log.setLevel(level=logging.DEBUG)
        from hpobench.util.container_utils import enable_container_debug
        enable_container_debug()

    from hpobench import config_file

    optimizer_settings = get_optimizer_setting(optimizer)
    benchmark_settings = get_benchmark_settings(benchmark)
    settings = dict(optimizer_settings, **benchmark_settings)
    settings['benchmark_params'] = benchmark_params

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

    resource_file_dir = resource_file_dir \
        if resource_file_dir is not None else os.environ.get('TMPDIR', '/tmp/')
    resource_file_dir = Path(resource_file_dir)
    if not resource_file_dir.exists():
        raise NotADirectoryError('The directory for the resource file does not exist. Please create it.'
                                 f'Given directory was: {resource_file_dir}')

    resource_file_dir = resource_file_dir / dname / optimizer / f'run-{rng}'
    resource_file_dir = resource_file_dir.expanduser().absolute()
    resource_file_dir.mkdir(exist_ok=True, parents=True)

    _log.debug(f'Output dir: {output_dir}. Resource file is in: {resource_file_dir}')

    # Load and instantiate the benchmark
    benchmark_obj = load_benchmark(benchmark_name=settings['import_benchmark'],
                                   import_from=settings['import_from'],
                                   use_local=use_local)

    if use_local:
        benchmark = benchmark_obj(rng=rng, **benchmark_params)
    else:
        container_source = config_file.container_source
        benchmark = benchmark_obj(rng=rng, container_source=container_source, **benchmark_params)
    _log.info(f'Benchmark successfully initialized.')

    start_time = time()

    process = Process(target=subprocess_run, args=(),
                      kwargs=dict(settings=settings, use_local=use_local, socket_id=benchmark.socket_id, rng=rng,
                                  optimizer_enum=optimizer_enum, output_dir=output_dir,
                                  resource_file_dir=resource_file_dir))
    process.run()

    resource_file = resource_file_dir / 'used_resources.json'
    lock_dir = output_dir / 'lock_dir'
    resource_lock_file = 'resource_lock'

    time_waited = 0
    while not resource_file.exists():
        if (round(time_waited, 2) % 2) == 0:
            _log.info('Wait for bookkeeper to come alive')
        sleep(0.1)
        time_waited += 0.1

        if time_waited > 10:
            raise TimeoutError('Cannot find the resource file. Something seems to be broken.')

    _log.info('Bookkeeper initizalized. Start Timer.')
    resources = Bookkeeper.load_resource_file(resource_file, lock_dir, resource_lock_file)

    while not total_time_exceeds_limit(resources['total_time_proxy'], settings['time_limit_in_s'], resources['start_time']) \
            and not tae_exceeds_limit(resources['total_tae_calls_proxy'], settings['tae_limit']) \
            and not used_fuel_exceeds_limit(resources['total_fuel_used_proxy'], settings['fuel_limit']) \
            and process.is_alive():
        sleep(PING_OPTIMIZER_IN_S)
        resources = Bookkeeper.load_resource_file(resource_file, lock_dir, resource_lock_file)

    else:
        _log.debug('CALLING TERMINATE()')
        process.terminate()
        _log.debug('PROCESS TERMINATED')
        _log.info(f'Optimization has been finished.\n'
                  f'Timelimit: {settings["time_limit_in_s"]} and is now: {resources["total_time_proxy"]}\n'
                  f'TAE limit: {settings["tae_limit"]} and is now: {resources["total_tae_calls_proxy"]}\n'
                  f'Fuel limit: {settings["fuel_limit"]} and is now: {resources["total_fuel_used_proxy"]}\n'
                  f'Terminate Process after {time() - start_time}')

    _log.info(f'Extract the trajectories')
    extract_trajectory(output_dir=output_dir, debug=debug)

    _log.info(f'Run Benchmark - Finished.')
    benchmark._shutdown()


def subprocess_run(settings, use_local, socket_id, rng, optimizer_enum, output_dir, resource_file_dir):
    _log.info(f'Subprocess called with params: setting:{settings}\n'
              f'use_local:{use_local}\nsocket_id{socket_id}\nrng:{rng}\noutput_dir:{output_dir}')

    from hpobench import config_file
    container_source = config_file.container_source

    benchmark_obj = load_benchmark(benchmark_name=settings['import_benchmark'],
                                   import_from=settings['import_from'],
                                   use_local=use_local)

    from functools import partial
    benchmark_partial = partial(benchmark_obj,
                                rng=rng,
                                container_source=container_source,
                                socket_id=socket_id,
                                **settings['benchmark_params'])

    book_keeper = Bookkeeper(benchmark_partial=benchmark_partial,
                             output_dir=output_dir,
                             resource_file_dir=resource_file_dir,
                             wall_clock_limit_in_s=settings['time_limit_in_s'],
                             tae_limit=settings['tae_limit'],
                             fuel_limit=settings['fuel_limit'],
                             cutoff_limit_in_s=settings['cutoff_in_s'],
                             is_surrogate=settings['is_surrogate'])

    _log.info(f'Bookkeeper initialized. Additional benchmark parameters {settings["benchmark_params"]}')

    optimizer = get_optimizer(optimizer_enum)
    optimizer = optimizer(benchmark=book_keeper,
                          settings=settings,
                          output_dir=output_dir,
                          rng=rng)
    _log.info(f'Optimizer initialized. Start optimization process.')

    # Currently no optimizer uses the setup function, but we still call it to enable future optimizer
    # implementations to have a setup function.
    optimizer.setup()
    optimizer.run()

    _log.info('Shutdown the Optimizer and the book keeper. ')
    try:
        optimizer.shutdown()
        book_keeper.final_shutdown()
    except Exception as e:
        # TODO: Test this here. But in case it crashes, ignore the error so that the trajectory extraction
        #       in the main process doesn't fail.
        _log.info('During the shutdown procedure, an error was raised')
        _log.exception(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='HPOBench Wrapper',
                                     description='HPOBench running different benchmarks on different optimizer with a '
                                                 'unified interface')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--resource_file_dir', required=False, type=str,
                        help='We track the currently used resources in a file. By default, this is the tmp directory.')
    parser.add_argument('--optimizer', choices=get_optimizer_settings_names(), required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)
    parser.add_argument('--use_local', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False, help="When given, enables debug mode logging.")
    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    run_benchmark(**vars(args), **benchmark_params)
