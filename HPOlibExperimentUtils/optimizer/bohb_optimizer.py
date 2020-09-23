import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Type

import ConfigSpace as CS
from hpbandster.core import result as hpres, nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB, HyperBand, RandomSearch, H2BO
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOlibExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOlibExperimentUtils.utils.optimizer_utils import get_main_fidelity

logger = logging.getLogger('Optimizer')


class HpBandSterBaseOptimizer(SingleFidelityOptimizer):
    """
    This class offers an interface to the BOHB Optimizer. It runs on a given benchmark.
    All benchmark and optimizer specific information are stored in the dictionaries benchmark_settings and
    optimizer_settings.
    """
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 intensifier: Union[Type[BOHB], Type[HyperBand], Type[H2BO], Type[RandomSearch]],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        self.run_id = f'BOHB_optimization_seed_{self.rng}'
        self.intensifier = intensifier

    def setup(self):
        pass

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """
        result_logger = hpres.json_result_logger(directory=str(self.output_dir), overwrite=True)

        ns = hpns.NameServer(run_id=self.run_id,
                             host='localhost',
                             working_directory=str(self.output_dir))
        ns_host, ns_port = ns.start()

        worker = CustomWorker(benchmark=self.benchmark,
                              main_fidelity=self.main_fidelity,
                              settings=self.settings,
                              settings_for_sending=self.settings_for_sending,
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              run_id=self.run_id)

        worker.run(background=True)

        master = self.intensifier(configspace=self.cs,
                                  run_id=self.run_id,
                                  host=ns_host,
                                  nameserver=ns_host,
                                  nameserver_port=ns_port,
                                  eta=self.settings['eta'],
                                  min_budget=self.min_budget,
                                  max_budget=self.max_budget,
                                  result_logger=result_logger)

        result = master.run(n_iterations=self.settings['num_iterations'])
        with open(self.output_dir / 'results.pkl', 'wb') as fh:
            pickle.dump(result, fh)

        master.shutdown(shutdown_workers=True)
        ns.shutdown()

        # Todo: We can recover the results from the master object.
        # result = master.run(n_iterations=self.settings['num_iterations'])
        # for iteration in master.warmstart_iteration:
        #     iteration.fix_timestamps(master.time_ref)
        # ws_data = [iteration.data for iteration in master.warmstart_iteration]
        # result = hpres.Result([deepcopy(iteration.data) for iteration in master.iterations] + ws_data,
        #                       master.config)

        # if result is not None:
        #     id2config = result.get_id2config_mapping()
        #     incumbent = result.get_incumbent_id()
        #     inc_value = result.get_runs_by_id(incumbent)[-1]['loss']
        #     inc_cfg = id2config[incumbent]['config']
        # logger.info(f'Inc Config:\n{inc_cfg}\n with Performance: {inc_value:.2f}')


class HpBandSterBOHBOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterBOHBOptimizer, self).__init__(benchmark=benchmark, intensifier=BOHB, settings=settings,

                                                      output_dir=output_dir, rng=rng)


class HpBandSterRandomSearchOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterRandomSearchOptimizer, self).__init__(benchmark=benchmark, intensifier=RandomSearch,
                                                              settings=settings, output_dir=output_dir, rng=rng)


class HpBandSterHyperBandOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterHyperBandOptimizer, self).__init__(benchmark=benchmark, intensifier=HyperBand,
                                                           settings=settings, output_dir=output_dir, rng=rng)


class HpBandSterH2BOOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterH2BOOptimizer, self).__init__(benchmark=benchmark, intensifier=H2BO,
                                                      settings=settings, output_dir=output_dir, rng=rng)


class CustomWorker(Worker):
    """ A generic worker for optimizing with BOHB. """
    def __init__(self, benchmark: AbstractBenchmark,
                 main_fidelity: CS.Configuration,
                 settings: Dict,
                 settings_for_sending: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.main_fidelity = main_fidelity
        self.settings = settings
        self.settings_for_sending = settings_for_sending

        self.main_fidelity = get_main_fidelity(fidelity_space=benchmark.get_fidelity_space(),
                                               settings=settings)

    def compute(self, config: Dict, budget: Any, **kwargs) -> Dict:
        """Here happens the work in the optimization step. """
        fidelity = {self.main_fidelity.name: budget}

        result_dict = self.benchmark.objective_function(configuration=config, fidelity=fidelity, **self.settings_for_sending)
        return {'loss': result_dict['function_value'],
                'info': result_dict}
