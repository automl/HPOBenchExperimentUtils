import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Type

import ConfigSpace as CS
from hpbandster.core import result as hpres, nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB, HyperBand, RandomSearch, H2BO
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.utils.optimizer_utils import get_main_fidelity

_log = logging.getLogger(__name__)


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

        # Hpbandster does not handle ordinal hyperparameter.
        # cast them to categorical.
        hps = self.cs.get_hyperparameters()

        ordinal_hp = any([isinstance(hp, CS.OrdinalHyperparameter) for hp in hps])

        if ordinal_hp:
            new_cs = CS.ConfigurationSpace()
            new_cs.seed(self.rng)
            for hp in hps:
                if isinstance(hp, CS.OrdinalHyperparameter):
                    values = [hp.get_value(index) for index in hp.get_seq_order()]
                    cat_hp = CS.CategoricalHyperparameter(hp.name, choices=values, default_value=hp.default_value)
                    new_cs.add_hyperparameter(cat_hp)
                    _log.info(f'Convert Ordinal Hyperparameter: {hp.name} to categorical.')
                else:
                    new_cs.add_hyperparameter(hp)
            self.cs = new_cs

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

        # Allow at most max_stages stages
        tmp = self.max_budget
        for i in range(self.settings.get('max_stages', 10)):
            tmp /= self.settings['eta']
        if tmp > self.min_budget:
            self.min_budget = tmp
        
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
        # _log.info(f'Inc Config:\n{inc_cfg}\n with Performance: {inc_value:.2f}')


class HpBandSterBOHBOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterBOHBOptimizer, self).__init__(benchmark=benchmark, intensifier=BOHB, settings=settings,
                                                      output_dir=output_dir, rng=rng)


class HpBandSterHyperBandOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterHyperBandOptimizer, self).__init__(benchmark=benchmark, intensifier=HyperBand,
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

        if isinstance(self.main_fidelity, CS.hyperparameters.UniformIntegerHyperparameter) \
                or isinstance(self.main_fidelity, CS.hyperparameters.NormalIntegerHyperparameter) \
                or isinstance(self.main_fidelity.default_value, int):
            budget = int(budget)
        fidelity = {self.main_fidelity.name: budget}

        result_dict = self.benchmark.objective_function(configuration=config,
                                                        fidelity=fidelity,
                                                        **self.settings_for_sending)
        return {'loss': result_dict['function_value'],
                'info': result_dict}
