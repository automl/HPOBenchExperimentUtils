import logging
from importlib import import_module
from pathlib import Path
from typing import Union, Dict

from trajectory_parser.optimizer import BOHBOptimizer, SMACOptimizer
from trajectory_parser.utils.runner_utils import transform_unknown_params_to_dict, get_setting_per_benchmark, \
    OptimizerEnum, optimizer_str_to_enum

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('BenchmarkRunner')


def run_benchmark(optimizer: Union[OptimizerEnum, str],
                  benchmark: str,
                  output_dir: Union[Path, str],
                  seed: int,
                  **benchmark_params: Dict):

    optimizer_enum = optimizer_str_to_enum(optimizer)

    output_dir = Path(output_dir) / f'{str(optimizer_enum)}-run-{seed}'
    output_dir.mkdir(exist_ok=True, parents=True)

    optimizer_settings, benchmark_settings = get_setting_per_benchmark(benchmark, seed=seed, output_dir=output_dir)

    # Load benchmark
    module = import_module(f'hpolib.benchmarks.{benchmark_settings["import_from"]}')
    benchmark_obj = getattr(module, benchmark_settings['import_benchmark'])

    benchmark = benchmark_obj(**benchmark_params)  # Todo: Arguments for Benchmark? --> b(**benchmark_params)

    # setup optimizer (either smac or bohb)
    optimizer = BOHBOptimizer if optimizer_enum is OptimizerEnum.BOHB else SMACOptimizer
    optimizer = optimizer(benchmark=benchmark, optimizer_settings=optimizer_settings,
                          benchmark_settings=benchmark_settings, intensifier=optimizer)
    # optimizer.setup()
    optimizer.run()

    logger.info('Runner finished')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 running different benchmarks on different optimizer with a '
                                                 'unified interface',
                                     usage='%(prog)s --output_dir <str> '
                                           '--optimizer [BOHB|SMAC|HYPERBAND|SUCCESSIVE_HALVING] '
                                           '--benchmark [xgboost|CartpoleFull|CartpoleReduced]'
                                           '--seed <int>'
                                           '[--benchmark_parameter1 value, ...]')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--optimizer', choices=['BOHB', 'SMAC', 'HYPERBAND', 'SUCCESSIVE_HALVING'], required=True,
                        type=str)
    parser.add_argument('--benchmark', required=True, type=str)
    parser.add_argument('--seed', required=False, default=0, type=int)

    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    run_benchmark(**vars(args), **benchmark_params)
