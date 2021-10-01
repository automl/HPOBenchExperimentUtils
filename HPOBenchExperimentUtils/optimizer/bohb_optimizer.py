import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Type

import ConfigSpace as CS
from hpbandster.core import result as hpres, nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB, HyperBand, RandomSearch, H2BO

from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.utils.optimizer_utils import get_main_fidelity

_log = logging.getLogger(__name__)


class HpBandSterBaseOptimizer(SingleFidelityOptimizer):
    """
    This class offers an interface to the BOHB Optimizer. It runs on a given benchmark.
    All benchmark and optimizer specific information are stored in the dictionaries benchmark_settings and
    optimizer_settings.
    """
    def __init__(self, benchmark: Bookkeeper,
                 intensifier: Union[Type[BOHB], Type[HyperBand], Type[H2BO], Type[RandomSearch]],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

        # Hpbandster does not handle ordinal hyperparameter. Cast them to integer.
        self.new_cs = CS.ConfigurationSpace()
        self.new_cs.seed(self.rng)

        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, CS.OrdinalHyperparameter):
                p = CS.UniformIntegerHyperparameter(hp.name, lower=0, upper=len(hp.sequence)-1)
                self.new_cs.add_hyperparameter(p)
            else:
                self.new_cs.add_hyperparameter(hp)
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
                              run_id=self.run_id,
                              orig_cs=self.cs,
                              )

        worker.run(background=True)

        # Allow at most max_stages stages
        tmp = self.max_budget
        for i in range(self.settings.get('max_stages', 10)):
            tmp /= self.settings.get('eta', 3)
        if tmp > self.min_budget:
            self.min_budget = tmp
        
        master = self.intensifier(configspace=self.new_cs,
                                  run_id=self.run_id,
                                  host=ns_host,
                                  nameserver=ns_host,
                                  nameserver_port=ns_port,
                                  eta=self.settings.get('eta', 3),
                                  min_budget=self.min_budget,
                                  max_budget=self.max_budget,
                                  result_logger=result_logger)

        result = master.run(n_iterations=self.settings['num_iterations'])
        ## We don't want to generate additional data
        #with open(self.output_dir / 'results.pkl', 'wb') as fh:
        #    pickle.dump(result, fh)

        master.shutdown(shutdown_workers=True)
        ns.shutdown()


class HpBandSterBOHBOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterBOHBOptimizer, self).__init__(benchmark=benchmark, intensifier=BOHB, settings=settings,
                                                      output_dir=output_dir, rng=rng)


class HpBandSterHyperBandOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterHyperBandOptimizer, self).__init__(benchmark=benchmark, intensifier=HyperBand,
                                                           settings=settings, output_dir=output_dir, rng=rng)


class HpBandSterTPEOptimizer(HpBandSterBaseOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super(HpBandSterTPEOptimizer, self).__init__(benchmark=benchmark, intensifier=BOHB,
                                                     settings=settings, output_dir=output_dir, rng=rng)
        self.min_budget = self.max_budget


class CustomWorker(Worker):
    """ A generic worker for optimizing with BOHB. """
    def __init__(self, benchmark: Bookkeeper,
                 main_fidelity: CS.Configuration,
                 settings: Dict,
                 settings_for_sending: Dict, orig_cs: CS.ConfigurationSpace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.main_fidelity = main_fidelity
        self.settings = settings
        self.settings_for_sending = settings_for_sending
        self.orig_cs = orig_cs
        self.main_fidelity = get_main_fidelity(fidelity_space=benchmark.get_fidelity_space(),
                                               settings=settings)

    def compute(self, config: Dict, budget: Any, **kwargs) -> Dict:
        """Here happens the work in the optimization step. """

        run_id = SingleFidelityOptimizer._id_generator()

        for h in self.orig_cs.get_hyperparameters():
            if isinstance(h, CS.OrdinalHyperparameter):
                config[h.name] = h.sequence[int(config[h.name])]

        if isinstance(self.main_fidelity, CS.hyperparameters.UniformIntegerHyperparameter) \
                or isinstance(self.main_fidelity, CS.hyperparameters.NormalIntegerHyperparameter) \
                or isinstance(self.main_fidelity.default_value, int):
            budget = int(budget)
        fidelity = {self.main_fidelity.name: budget}

        result_dict = self.benchmark.objective_function(configuration=config,
                                                        configuration_id=run_id,
                                                        fidelity=fidelity,
                                                        **self.settings_for_sending)
        return {'loss': result_dict['function_value'],
                'info': result_dict}
