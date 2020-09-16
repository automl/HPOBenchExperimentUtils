import logging
from multiprocessing import Process, Manager
from pathlib import Path
from time import time, sleep
from typing import Union, Dict

from hpolib.util.container_utils import enable_container_debug
from hpolib.util.example_utils import set_env_variables_to_use_only_one_core

from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from HPOlibExperimentUtils.utils import PING_OPTIMIZER_IN_S
from HPOlibExperimentUtils.utils.optimizer_utils import get_optimizer
from HPOlibExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
    OptimizerEnum, optimizer_str_to_enum, load_benchmark, get_benchmark_names, get_optimizer_settings_names, \
    get_optimizer_setting

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('BenchmarkRunner')

set_env_variables_to_use_only_one_core()
enable_container_debug()


def run_benchmark(optimizer: Union[OptimizerEnum, str],
                  benchmark: str,
                  output_dir: Union[Path, str],
                  rng: int,
                  use_local: Union[bool, None] = False,
                  **benchmark_params: Dict):
    """
    Run a HPOlib3 benchmark on a given Optimizer. Currently only SMAC, BOHB and Dragonfly are available as Optimizer.
    The benchmarks are by default stored in singularity container which are downloaded at the first run.

    Parameters
    ----------
    optimizer : str, OptimizerEnum
        Either the OptimizerEnum object specifying the optimizer to take or a string representation of it.
        Allowed choices are:
        'BOHB',
        'HYPERBAND', or  'HB',
        'SUCCESSIVE_HALVING', or 'SH',
        'DRAGONFLY' or 'DF'
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

    logger.info(f'Start running benchmark.')

    optimizer_settings = get_optimizer_setting(optimizer)
    benchmark_settings = get_benchmark_settings(benchmark)
    settings = dict(benchmark_settings, **optimizer_settings)
    logger.debug(f'Settings loaded')

    optimizer_enum = optimizer_str_to_enum(optimizer)
    logger.debug(f'Optimizer: {optimizer_enum}')

    output_dir = Path(output_dir) / f'{str(optimizer_enum)}-run-{rng}'
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f'Output dir: {output_dir}')

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

    # Create a Process Manager to get access to the variable "total_time_proxy" of the bookkeeper
    # This variable represents how much time the optimizer has used
    manager = Manager()
    total_time_proxy = manager.Value('f', 0)

    benchmark = Bookkeeper(benchmark,
                           output_dir,
                           total_time_proxy,
                           wall_clock_limit_in_s=settings['time_limit_in_s'],
                           cutoff_limit_in_s=settings['cutoff_in_s'],
                           is_surrogate=settings['is_surrogate'])

    logger.debug(f'Benchmark initialized. Additional benchmark parameters {benchmark_params}')

    optimizer = get_optimizer(optimizer_enum)
    optimizer = optimizer(benchmark=benchmark,
                          settings=settings,
                          output_dir=output_dir, rng=rng)
    logger.debug(f'Optimizer initialized')

    start_time = time()
    process = Process(target=optimizer.run, args=(), kwargs=dict())
    process.start()

    # Wait for the optimizer to finish. But in case the optimizer crashes somehow, also test for the real time here.
    while settings['time_limit_in_s'] >= benchmark.get_total_time_used() \
            and settings['time_limit_in_s'] >= (time() - start_time) \
            and process.is_alive():
        sleep(PING_OPTIMIZER_IN_S)
    else:
        process.terminate()
        print(f'finished after {time() - start_time}')

    logger.info(f'Optimizer finished')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 running different benchmarks on different optimizer with a '
                                                 'unified interface')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--optimizer', choices=get_optimizer_settings_names(),  required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)

    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    run_benchmark(**vars(args), **benchmark_params)
