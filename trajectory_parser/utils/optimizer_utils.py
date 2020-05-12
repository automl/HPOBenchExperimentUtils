import logging

from hpbandster.core.worker import Worker

from trajectory_parser.utils.utils import TimeoutException, time_limit

logger = logging.getLogger('Optimizer Utils')


class CustomWorker(Worker):
    def __init__(self, benchmark, benchmark_settings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark_settings = benchmark_settings

    def compute(self, config, budget, **kwargs):
        fidelity = {self.benchmark_settings['fidelity_name']: self.benchmark_settings['fidelity_type'](budget)}

        try:
            with time_limit(self.benchmark_settings['cutoff_in_s']):
                result_dict = self.benchmark.objective_function(config, **fidelity, **self.benchmark_settings)
        except TimeoutException:
            raise TimeoutError('Cutoff time reached.')

        return {'loss': result_dict['function_value'],
                # TODO: add result dict in a generic fashion with also "non-pickable" return types.
                'info': {k: v for k, v in result_dict.items()}
                }