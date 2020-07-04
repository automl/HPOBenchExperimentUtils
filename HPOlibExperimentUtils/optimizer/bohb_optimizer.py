import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from hpbandster.core import result as hpres, nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from hpolib.abstract_benchmark import AbstractBenchmark

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOlibExperimentUtils.utils.optimizer_utils import parse_fidelity_type
from HPOlibExperimentUtils.utils.utils import TimeoutException, time_limit

logger = logging.getLogger('Optimizer')


class BOHBOptimizer(Optimizer):
    def __init__(self, benchmark, optimizer_settings, benchmark_settings, intensifier, rng=0):
        super().__init__(benchmark, optimizer_settings, benchmark_settings, intensifier, rng)
        self.run_id = f'BOHB_optimization_seed_{self.rng}'

    def setup(self):
        pass

    def run(self) -> Path:
        result_logger = hpres.json_result_logger(directory=str(self.optimizer_settings['output_dir']), overwrite=True)

        ns = hpns.NameServer(run_id=self.run_id,
                             host='localhost',
                             working_directory=str(self.optimizer_settings['output_dir']))
        ns_host, ns_port = ns.start()

        worker = CustomWorker(benchmark=self.benchmark,
                              benchmark_settings=self.benchmark_settings,
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              run_id=self.run_id)

        worker.run(background=True)

        master = BOHB(configspace=self.cs,
                      run_id=self.run_id,
                      host=ns_host,
                      nameserver=ns_host,
                      nameserver_port=ns_port,
                      eta=self.optimizer_settings['eta'],
                      min_budget=self.optimizer_settings['min_budget'],
                      max_budget=self.optimizer_settings['max_budget'],
                      result_logger=result_logger)

        # TODO: Do it the same way as in smac: try with finally
        try:
            with time_limit(self.optimizer_settings['time_limit_in_s']):
                result = master.run(n_iterations=self.optimizer_settings['num_iterations'])
        except TimeoutException:
            for iteration in master.warmstart_iteration:
                iteration.fix_timestamps(master.time_ref)
            ws_data = [iteration.data for iteration in master.warmstart_iteration]
            result = hpres.Result([deepcopy(iteration.data) for iteration in master.iterations] + ws_data,
                                  master.config)
            logger.info('WALLCLOCK LIMIT REACHED')

        master.shutdown(shutdown_workers=True)
        ns.shutdown()

        if result is not None:
            id2config = result.get_id2config_mapping()
            incumbent = result.get_incumbent_id()
            inc_value = result.get_runs_by_id(incumbent)[-1]['loss']
            inc_cfg = id2config[incumbent]['config']

            logger.info(f'Inc Config:\n{inc_cfg}\n with Performance: {inc_value:.2f}')

        return self.optimizer_settings['output_dir']


class CustomWorker(Worker):
    """ A generic worker for optimizing with BOHB. """
    def __init__(self, benchmark : AbstractBenchmark, benchmark_settings: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark_settings = benchmark_settings

    def compute(self, config: Dict, budget: Any, **kwargs) -> Dict:

        fidelity_type = parse_fidelity_type(self.benchmark_settings['fidelity_type'])
        fidelity = {self.benchmark_settings['fidelity_name']: fidelity_type(budget)}

        result_dict = self.benchmark.objective_function(configuration=config, **fidelity, **self.benchmark_settings)
        return {'loss': result_dict['function_value'],
                # TODO: add result dict in a generic fashion with also "non-pickable" return types.
                'info': {k: v for k, v in result_dict.items()}
                }
