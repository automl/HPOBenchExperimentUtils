import ConfigSpace as CS
import os

from ConfigSpace import ConfigurationSpace
from ray.tune.utils.log import Verbosity
from ray import tune as tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter, BasicVariantGenerator
import ray

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
import ConfigSpace.hyperparameters as CSH

from typing import Union, Dict, Any
from pathlib import Path

socket_id = None
from functools import partial


class RayBaseOptimizer(SingleFidelityOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 search_algorithm: Any,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):

        super(RayBaseOptimizer, self).__init__(benchmark, settings, output_dir, rng)

        # We need to cast the ConfigSpace.configuration space to a ray configuration space
        self.cs_ray = configspace_to_ray_cs(benchmark.get_configuration_space(rng))

        if not isinstance(search_algorithm, BasicVariantGenerator):
            search_algorithm = ConcurrencyLimiter(search_algorithm, max_concurrent=1)

        self.search_algorithm = search_algorithm

        self.run_id = f'Ray_optimization_{search_algorithm}_seed_{rng}'

        self.valid_budgets = None
        self.scheduler = None

    def setup(self):
        pass

    def run(self):
        raise NotImplementedError()

    @staticmethod
    def _setup_ray():
        ray.init(local_mode=True,
                 log_to_driver=False,
                 include_dashboard=False)

    @staticmethod
    def _training_function(config, benchmark, main_fidelity_name, valid_budgets, configspace: ConfigurationSpace):

        from ray.tune import get_trial_id
        id = get_trial_id()

        print(id)
        print(f'IS ID NONE:{id is None}')

        run_id = SingleFidelityOptimizer._id_generator()
        config = fix_config_data_types(config, configspace)
        for budget in valid_budgets:
            result_dict = benchmark.objective_function(configuration=config,
                                                       configuration_id=run_id,
                                                       fidelity={main_fidelity_name: budget})
            tune.report(function_value=result_dict['function_value'], fidelity=budget)


class RayOptimizerWithoutFidelity(RayBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 search_algorithm: Any,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):

        super(RayOptimizerWithoutFidelity, self).__init__(benchmark, search_algorithm, settings, output_dir, rng)

    def run(self):
        self._setup_ray()

        # For random-search, evaluate only the highest fidelity.
        self.valid_budgets = [self.max_budget]

        tmp_dir = os.environ.get('TMPDIR', '/tmp/')

        tune.run(partial(self._training_function,
                         benchmark=self.benchmark,
                         main_fidelity_name=self.main_fidelity.name,
                         valid_budgets=self.valid_budgets,
                         configspace=self.cs),
                 metric='function_value',
                 mode='min',

                 # stop=lambda trial_id, results: False,
                 verbose=Verbosity.V1_EXPERIMENT,
                 config=self.cs_ray,
                 search_alg=self.search_algorithm,
                 # scheduler=ray.tune.schedulers.FIFOScheduler(),  # FiFO is default scheduler
                 resources_per_trial={"cpu": 1, "gpu": 0},
                 local_dir=tmp_dir,
                 # Set this to a very large number, so that this process runs is not bounded by the number of samples.
                 num_samples=100000000000)


class RayRandomSearchOptimizer(RayOptimizerWithoutFidelity):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):

        from ray.tune.suggest.basic_variant import BasicVariantGenerator
        search_algorithm = BasicVariantGenerator(max_concurrent=1)
        super(RayRandomSearchOptimizer, self).__init__(benchmark, search_algorithm, settings, output_dir, rng)


class RayHyperoptWithoutFidelityOptimizer(RayOptimizerWithoutFidelity):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):

        from ray.tune.suggest.hyperopt import HyperOptSearch
        search_algorithm = HyperOptSearch()
        super(RayHyperoptWithoutFidelityOptimizer, self).\
            __init__(benchmark, search_algorithm, settings, output_dir, rng)


class RayHBBaseOptimizer(RayBaseOptimizer):
    """
    A base class for ray optimizer. Ray implements several optimizer, e.g. the hyperopt optimizer.
    We use Hyperband as scheduler.

    To install only ray without a optimizer:
    pip install <path to the HPOBenchExperimentUtils>[ray_base]

    For Hyperopt:
    pip install <path to the HPOBenchExperimentUtils>[ray_base,ray_hyperopt]

    For Bayesopt:
    pip install <path to the HPOBenchExperimentUtils>[ray_base,ray_bayesopt]

    For Optuna:
    pip install <path to the HPOBenchExperimentUtils>[ray_base,ray_optuna]

    IMPORTANT:
    ==========
    Disable the default logging of ray. (Especially for the surrogate benchmarks.)
    Set the env variable `TUNE_DISABLE_AUTO_CALLBACK_LOGGERS` to 1
    (os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = 1)
    """

    def __init__(self,
                 benchmark: Bookkeeper,
                 search_algorithm: Any,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(RayHBBaseOptimizer, self).__init__(benchmark, search_algorithm, settings, output_dir, rng)

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """
        self._setup_ray()

        # We use here a pure succesive halving. No Hyperband
        self.scheduler = AsyncHyperBandScheduler(max_t=self.max_budget,
                                                 grace_period=self.min_budget,
                                                 reduction_factor=self.settings['reduction_factor'],
                                                 time_attr='fidelity',
                                                 brackets=1)

        self.valid_budgets = [self.max_budget] + [budget for (budget, _) in self.scheduler._brackets[0]._rungs]
        self.valid_budgets = list(set(self.valid_budgets))
        self.valid_budgets.sort()

        tmp_dir = os.environ.get('TMPDIR', '/tmp/')

        tune.run(partial(self._training_function,
                         benchmark=self.benchmark,
                         main_fidelity_name=self.main_fidelity.name,
                         valid_budgets=self.valid_budgets,
                         configspace=self.cs,
                         ),
                 metric='function_value',
                 mode='min',
                 verbose=Verbosity.V1_EXPERIMENT,
                 config=self.cs_ray,
                 search_alg=self.search_algorithm,
                 scheduler=self.scheduler,
                 resources_per_trial={"cpu": 1, "gpu": 0},
                 local_dir=tmp_dir,
                 # Set this to a very large number, so that this process runs is not bounded by the number of samples.
                 num_samples=10000000,
                 )


class RayHBHyperoptOptimizer(RayHBBaseOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        from ray.tune.suggest.hyperopt import HyperOptSearch

        search_algorithm = HyperOptSearch()

        super(RayHBHyperoptOptimizer, self).__init__(benchmark=benchmark,
                                                     search_algorithm=search_algorithm,
                                                     settings=settings,
                                                     output_dir=output_dir, rng=rng)


class RayHBBayesOptOptimizer(RayHBBaseOptimizer):
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        from ray.tune.suggest.bayesopt import BayesOptSearch

        search_algorithm = BayesOptSearch()

        super(RayHBBayesOptOptimizer, self).__init__(benchmark=benchmark,
                                                     search_algorithm=search_algorithm,
                                                     settings=settings,
                                                     output_dir=output_dir, rng=rng)


def configspace_to_ray_cs(cs: ConfigurationSpace):
    ray_cs = {}
    for hp_name in cs:
        hp = cs.get_hyperparameter(hp_name)
        ray_hp = None
        import numpy as np

        if isinstance(hp, CS.UniformFloatHyperparameter):
            if hp.log:
                ray_hp = tune.loguniform(hp.lower, hp.upper, base=np.e)
            else:
                ray_hp = tune.uniform(hp.lower, hp.upper)

        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            if hp.log:
                ray_hp = tune.qloguniform(hp.lower, hp.upper, q=1.0, base=np.e)
            else:
                ray_hp = tune.quniform(hp.lower, hp.upper, q=1.0)

        elif isinstance(hp, CS.CategoricalHyperparameter):
            ray_hp = tune.choice(hp.choices)

        elif isinstance(hp, CS.OrdinalHyperparameter):
            ray_hp = tune.quniform(0, len(hp.sequence)-1, q=1.0)

        ray_cs[hp.name] = ray_hp
    return ray_cs


def fix_config_data_types(configuration: Dict, configuration_space: CS.ConfigurationSpace) -> Dict:
    # Cast the configuration into the correct form
    new_config = {}
    for hp_name, value in configuration.items():
        hp = configuration_space.get_hyperparameter(hp_name)
        new_value = None
        if isinstance(hp, CS.UniformIntegerHyperparameter):
            new_value = int(value)
        elif isinstance(hp, CS.OrdinalHyperparameter):
            new_value = hp.sequence[int(value)]
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            new_value = float(value)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            hp_type = type(hp.default_value)
            new_value = hp_type(value)
        else:
            print(f'Unknown type of hyperparameter. type{value}')

        new_config[hp_name] = new_value or value
    config = new_config
    return config


__all__ = [RayHyperoptWithoutFidelityOptimizer, RayRandomSearchOptimizer,
           RayHBBayesOptOptimizer, RayHBHyperoptOptimizer]

#
# if __name__ == '__main__':
#     cs = ConfigurationSpace()
#     cs.add_hyperparameters([
#         CS.UniformFloatHyperparameter('eta', lower=2 ** -10, upper=1., default_value=0.3, log=True),
#         CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=15, default_value=6, log=True),
#         CS.CategoricalHyperparameter('test', choices=[1,2,3])
#     ])
#
#     configspace_to_ray_cs(cs)
