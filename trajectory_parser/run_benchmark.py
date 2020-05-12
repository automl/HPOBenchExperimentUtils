import logging
from importlib import import_module
from pathlib import Path
from typing import Union, Dict

from trajectory_parser.utils.optimizer import OptimizerEnum, optimizer_str_to_enum, BOHBOptimizer, SMACOptimizer
from trajectory_parser.utils.runner_utils import transform_unknown_params_to_dict

logger = logging.getLogger('BenchmarkRunner')


def run_benchmark(optimizer: Union[OptimizerEnum, str],
                  benchmark: str,
                  output_dir: Union[Path, str],
                  seed: int,
                  time_limit_in_s: int,
                  cutoff_in_s: int,
                  min_budget: Union[int, float],
                  max_budget: Union[int, float],
                  eta: int,
                  num_iterations: int,
                  **benchmark_params: Dict):

    output_dir = Path(output_dir) / f'run-{seed}'
    output_dir.mkdir(exist_ok=True, parents=True)
    from trajectory_parser.utils.runner_utils import get_setting_per_benchmark

    optimizer_settings, benchmark_settings = get_setting_per_benchmark(benchmark)

    custom_optimizer_settings = {'output_dir': output_dir,
                                 'seed': seed,
                                 'min_budget': min_budget,
                                 'max_budget': max_budget,
                                 'eta': eta,
                                 'num_iterations': num_iterations,
                                 'time_limit_in_s': time_limit_in_s,
                                 'cutoff_in_s': cutoff_in_s,
                                 }
    optimizer_settings.update(custom_optimizer_settings)

    # Load benchmark
    benchmark = benchmark.split('.')
    import_from, benchmark_name = f'{benchmark[0]}.{benchmark[1]}', benchmark[2]

    module = import_module(f'hpolib.benchmarks.{import_from}')
    benchmark_obj = getattr(module, benchmark_name)

    benchmark = benchmark_obj(**benchmark_params)  # Todo: Arguments for Benchmark? --> b(**benchmark_params)

    # setup optimizer (either smac or bohb)
    optimizer_enum = optimizer_str_to_enum(optimizer)
    optimizer = BOHBOptimizer if optimizer_enum is OptimizerEnum.BOHB else SMACOptimizer
    optimizer = optimizer(benchmark=benchmark, settings=optimizer_settings, benchmark_settings=benchmark_settings,
                          intensifier=optimizer)
    # optimizer.setup()
    optimizer.run()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 running different benchmarks on different optimizer with a '
                                                 'unified interface',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--output_dir', default='./cartpole_smac_hb', type=str)
    parser.add_argument('--optimizer', choices=['BOHB', 'SMAC'], required=True, type=str)
    parser.add_argument('--benchmark', required=True, type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--time_limit_in_s', default=3600, type=int)
    parser.add_argument('--cutoff_in_s', default=None, type=int)
    parser.add_argument('--min_budget', default=None, required=False, type=int)
    parser.add_argument('--max_budget', default=None, required=False, type=int)
    parser.add_argument('--eta', default=None, required=False, type=int)
    parser.add_argument('--num_iterations', default=None, required=False, type=int)

    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    # optimizer = OptimizerEnum.BOHB
    # args.benchmark = 'rl.cartpole.CartpoleReduced'
    # benchmark = 'ml.xgboost_benchmark.XGBoostBenchmark'
    run_benchmark(**vars(args), **benchmark_params)