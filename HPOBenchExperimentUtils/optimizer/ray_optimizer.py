import ConfigSpace as CS
from ConfigSpace import ConfigurationSpace
from ray.tune.utils.log import Verbosity
from ray import tune as tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from typing import Union, Dict, Any
from pathlib import Path

socket_id = None
from functools import partial


class RayBaseOptimizer(SingleFidelityOptimizer):
    """
    A base class for ray optimizer. Ray implements several optimizer, e.g. the hyperopt optimizer.
    We use Hyperband as scheduler.

    To install ray:
    pip install <path to the HPOBenchExperimentUtils>[ray]


    """

    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 search_algorithm: Any,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

        # We need to cast the ConfigSpace.configuration space to a ray configuration space
        self.cs_ray = configspace_to_ray_cs(self.benchmark.get_configuration_space(rng))

        self.run_id = f'Ray_optimization_{search_algorithm}_seed_{self.rng}'
        self.search_algorithm = search_algorithm
        self.scheduler = AsyncHyperBandScheduler(max_t=self.max_budget,
                                                 grace_period=self.min_budget,
                                                 reduction_factor=settings['reduction_factor'],
                                                 time_attr='fidelity',
                                                 brackets=1,
                                                 )
        self.valid_budgets = [self.max_budget] + [budget for (budget, _) in self.scheduler._brackets[0]._rungs]
        self.valid_budgets = list(set(self.valid_budgets))
        self.valid_budgets.sort()

    @staticmethod
    def __training_function(config, benchmark, main_fidelity_name, valid_budgets, configspace: ConfigurationSpace):

        from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter, \
            OrdinalHyperparameter

        # Cast the configuration into the correct form
        new_config = {}
        for hp_name, value in config.items():
            hp = configspace.get_hyperparameter(hp_name)
            new_value = None
            if isinstance(hp, (IntegerHyperparameter, OrdinalHyperparameter)):
                new_value = int(value)
            elif isinstance(hp, FloatHyperparameter):
                new_value = float(value)
            elif isinstance(hp, CategoricalHyperparameter):
                hp_type = type(hp.default_value)
                new_value = hp_type(value)
            else:
                print(f'Unknown type of hyperparameter. type{value}')

            new_config[hp_name] = new_value or value
        config = new_config

        for budget in valid_budgets:
            result_dict = benchmark.objective_function(config, fidelity={main_fidelity_name: budget})
            tune.report(function_value=result_dict['function_value'], fidelity=budget)

    def setup(self):
        pass

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """

        tune.run(partial(self.__training_function,
                         benchmark=self.benchmark,
                         main_fidelity_name=self.main_fidelity.name,
                         valid_budgets=self.valid_budgets,
                         configspace=self.cs,
                         ),
                 metric='function_value',
                 mode='min',

                 # stop=lambda trial_id, results: False,
                 verbose=Verbosity.V1_EXPERIMENT,
                 config=self.cs_ray,
                 search_alg=self.search_algorithm,
                 scheduler=self.scheduler,
                 resources_per_trial={"cpu": 1, "gpu": 0},
                 local_dir=str(self.output_dir),
                 # Set this to a very large number, so that this process runs is not bounded by the number of samples.
                 num_samples=10000000,
                 )


class RayHyperoptOptimizer(RayBaseOptimizer):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        from ray.tune.suggest.hyperopt import HyperOptSearch

        search_algorithm = HyperOptSearch()

        super(RayHyperoptOptimizer, self).__init__(benchmark=benchmark,
                                                   search_algorithm=search_algorithm,
                                                   settings=settings,
                                                   output_dir=output_dir, rng=rng)


def configspace_to_ray_cs(cs: ConfigurationSpace):
    ray_cs = {}
    for hp_name in cs:
        hp = cs.get_hyperparameter(hp_name)
        ray_hp = None

        if isinstance(hp, CS.UniformFloatHyperparameter):
            ray_hp = tune.uniform(hp.lower, hp.upper)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            ray_hp = tune.quniform(hp.lower, hp.upper, 1.0)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            ray_hp = tune.choice(hp.choices)
        ray_cs[hp.name] = ray_hp
    return ray_cs


if __name__ == '__main__':
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        CS.UniformFloatHyperparameter('eta', lower=2 ** -10, upper=1., default_value=0.3, log=True),
        CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=15, default_value=6, log=False),
        CS.CategoricalHyperparameter('test', choices=[1,2,3])
    ])

    configspace_to_ray_cs(cs)
