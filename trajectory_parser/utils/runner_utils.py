from typing import List

from hpbandster.core.worker import Worker


class CustomWorker(Worker):
    def __init__(self, benchmark, benchmark_settings, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark_settings = benchmark_settings
        self.seed = seed

    def compute(self, config, budget, **kwargs):
        result_dict = self.benchmark.objective_function(config, budget=int(budget), seed=self.seed,
                                                        **self.benchmark_settings)
        return {'loss': result_dict['function_value'],
                'info': {k: v for k, v in result_dict.items()}  # TODO: add result dict in a generic fashion
                }


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


def get_setting_per_benchmark(benchmark: str):
    optimizer_settings = {}
    benchmark_settings = {}

    if benchmark == 'rl.cartpole.CartpoleReduced' or benchmark == 'rl.cartpole.CartpoleFull':
        optimizer_settings = {'min_budget': 1,
                              'max_budget': 9,
                              'num_iterations': 10,
                              'time_limit_in_s': 4000,
                              'cutoff_in_s': 1800,
                              'mem_limit_in_mb': 4000,
                              }

    elif 'ml.xgboost_benchmark.XGBoostBenchmark' in benchmark:
        optimizer_settings = {'min_budget': 0.1,
                              'max_budget': 1.0,
                              'num_iterations': 10,
                              'time_limit_in_s': 4000,
                              'cutoff_in_s': 1800,
                              'mem_limit_in_mb': 4000,
                              }
        benchmark_settings = {'n_estimators': 128}
    else:
        optimizer_settings = {}

    return optimizer_settings, benchmark_settings