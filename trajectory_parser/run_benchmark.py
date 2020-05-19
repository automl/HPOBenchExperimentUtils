import logging
from importlib import import_module
from pathlib import Path
from typing import Union, Dict

from trajectory_parser import BOHBOptimizer, SMACOptimizer, BOHBReader, SMACReader
from trajectory_parser.utils.runner_utils import transform_unknown_params_to_dict, get_setting_per_benchmark, \
    OptimizerEnum, optimizer_str_to_enum

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('BenchmarkRunner')


def run_benchmark(optimizer: Union[OptimizerEnum, str],
                  benchmark: str,
                  output_dir: Union[Path, str],
                  rng: int,
                  **benchmark_params: Dict):

    optimizer_enum = optimizer_str_to_enum(optimizer)

    output_dir = Path(output_dir) / f'{str(optimizer_enum)}-run-{rng}'
    output_dir.mkdir(exist_ok=True, parents=True)

    optimizer_settings, benchmark_settings = get_setting_per_benchmark(benchmark, rng=rng, output_dir=output_dir)

    # Load benchmark
    module = import_module(f'hpolib.benchmarks.{benchmark_settings["import_from"]}')
    benchmark_obj = getattr(module, benchmark_settings['import_benchmark'])

    benchmark = benchmark_obj(**benchmark_params)

    # Setup optimizer (either smac or bohb)
    optimizer = BOHBOptimizer if optimizer_enum is OptimizerEnum.BOHB else SMACOptimizer
    optimizer = optimizer(benchmark=benchmark, optimizer_settings=optimizer_settings,
                          benchmark_settings=benchmark_settings, intensifier=optimizer_enum,
                          rng=rng)
    # optimizer.setup()
    run_dir = optimizer.run()

    # Export the trajectory
    traj_path = output_dir / f'traj_hpolib.json'
    logger.info('Runner finished')

    reader = BOHBReader() if optimizer_enum is OptimizerEnum.BOHB else SMACReader()
    reader.read(file_path=run_dir)
    reader.get_trajectory()
    reader.export_trajectory(traj_path)
    logger.info(f'Trajectory successfully exported to {traj_path}')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 running different benchmarks on different optimizer with a '
                                                 'unified interface',
                                     usage='%(prog)s --output_dir <str> '
                                           '--optimizer [BOHB|HYPERBAND|SUCCESSIVE_HALVING] '
                                           '--benchmark [xgboost|CartpoleFull|CartpoleReduced]'
                                           '--rng <int>'
                                           '[--benchmark_parameter1 value, ...]')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--optimizer', choices=['BOHB', 'HYPERBAND', 'HB', 'SUCCESSIVE_HALVING', 'SH'], required=True,
                        type=str)
    parser.add_argument('--benchmark', required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)

    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    run_benchmark(**vars(args), **benchmark_params)
