import logging
import numpy as np
from functools import partial
from pathlib import Path
from typing import Union, Dict, List

import ConfigSpace as CS
import optuna
from ConfigSpace import ConfigurationSpace
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.trial import Trial

from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaBaseOptimizer(SingleFidelityOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):

        super(OptunaBaseOptimizer, self).__init__(benchmark, settings, output_dir, rng)
        self.sampler = None
        self.pruner = None

    def setup(self):
        pass

    def run(self):
        raise NotImplementedError()

    @staticmethod
    def _training_function(config, benchmark, main_fidelity_name, valid_budgets, configspace: ConfigurationSpace):
        pass


class OptunaRandomSearchOptimizer(OptunaBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(OptunaRandomSearchOptimizer, self).__init__(benchmark, settings, output_dir, rng)
        self.sampler = RandomSampler(seed=rng)

    def run(self):
        sampler_class = self.sampler.__class__.__name__ or "None"
        logger.debug(f'Start Study with sampler {sampler_class}')

        study = optuna.create_study(direction='minimize',
                                    sampler=self.sampler)

        # noinspection PyTypeChecker
        study.optimize(func=partial(self.objective,
                                    benchmark=self.benchmark,
                                    main_fidelity_name=self.main_fidelity.name,
                                    max_budget=self.max_budget,
                                    configspace=self.cs),
                       timeout=None, n_trials=None)  # Run the optimization without a limitation

    @staticmethod
    def objective(trial: Trial, benchmark: Bookkeeper, main_fidelity_name: str, max_budget: Union[int, float],
                  configspace: ConfigurationSpace):

        configuration = sample_config_from_optuna(trial, configspace)

        run_id = SingleFidelityOptimizer._id_generator()
        result_dict = benchmark.objective_function(run_id, configuration, {main_fidelity_name: max_budget})

        trial.report(result_dict['function_value'], step=max_budget)
        return result_dict['function_value']


class OptunaBudgetBaseOptimizer(OptunaBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(OptunaBudgetBaseOptimizer, self).__init__(benchmark, settings, output_dir, rng)
        assert 'reduction_factor' in settings
        self.pruner = None
        self.valid_budgets = None

    def run(self):
        sampler_class = self.sampler.__class__.__name__ or "None"
        pruner_class = self.sampler.__class__.__name__ or "None"
        logger.debug(f'Start Study with sampler {sampler_class} and pruner {pruner_class}')

        study = optuna.create_study(direction='minimize',
                                    sampler=self.sampler,
                                    pruner=self.pruner)
        assert self.valid_budgets is not None and len(self.valid_budgets) != 0

        # noinspection PyTypeChecker
        study.optimize(func=partial(self.objective,
                                    benchmark=self.benchmark,
                                    main_fidelity_name=self.main_fidelity.name,
                                    valid_budgets=self.valid_budgets,
                                    configspace=self.cs),
                       timeout=None, n_trials=None)  # Run the optimization without a limitation

    @staticmethod
    def objective(trial: Trial, benchmark: Bookkeeper, main_fidelity_name: str, valid_budgets: List,
                  configspace: ConfigurationSpace):

        configuration = sample_config_from_optuna(trial, configspace)

        run_id = SingleFidelityOptimizer._id_generator()

        result_dict = None
        for budget in valid_budgets:
            result_dict = benchmark.objective_function(run_id, configuration, {main_fidelity_name: budget})
            trial.report(result_dict['function_value'], step=budget)

            if trial.should_prune():
                raise optuna.TrialPruned()

        assert result_dict is not None
        return result_dict['function_value']


class OptunaHyperbandBaseOptimizer(OptunaBudgetBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(OptunaHyperbandBaseOptimizer, self).__init__(benchmark, settings, output_dir, rng)
        reduction_factor = settings['reduction_factor']
        self.pruner = HyperbandPruner(min_resource=self.min_budget,
                                      max_resource=self.max_budget,
                                      reduction_factor=reduction_factor)

        # noinspection PyTypeChecker
        self.pruner._try_initialization(study=None)
        # self.trials_per_budget = self.pruner._trial_allocation_budgets
        self.valid_budgets = [self.min_budget * reduction_factor ** i for i in range(self.pruner._n_brackets)]


class OptunaTPEHyperbandOptimizer(OptunaHyperbandBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(OptunaTPEHyperbandOptimizer, self).__init__(benchmark, settings, output_dir, rng)

        self.sampler = TPESampler(seed=rng)


class OptunaCMAESHyperBandOptimizer(OptunaHyperbandBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(OptunaCMAESHyperBandOptimizer, self).__init__(benchmark, settings, output_dir, rng)

        self.sampler = CmaEsSampler(seed=rng)

        if any((isinstance(hp, CS.OrdinalHyperparameter) for hp in self.cs.get_hyperparameters())):
            logger.exception(f'Configuration Space with categorical hyperparameter: {self.cs}')
            raise ValueError('The CMA-ES Sampler only supports benchmarks without categorical hyperparameter.')


class OptunaTPEMedianStoppingOptimizer(OptunaBudgetBaseOptimizer):
    def __init__(self,
                 benchmark: Bookkeeper,
                 settings: Dict,
                 output_dir: Path,
                 rng: Union[int, None] = 0):
        super(OptunaTPEMedianStoppingOptimizer, self).__init__(benchmark, settings, output_dir, rng)
        reduction_factor = settings['reduction_factor']

        self.sampler = TPESampler(seed=rng)
        self.pruner = MedianPruner()

        sh_iters = precompute_sh_iters(self.min_budget, self.max_budget, reduction_factor)
        self.valid_budgets = precompute_budgets(self.max_budget, reduction_factor, sh_iters)


def sample_config_from_optuna(trial: Trial, cs: CS.ConfigurationSpace):

    config = {}
    for hp_name in cs:
        hp = cs.get_hyperparameter(hp_name)

        if isinstance(hp, CS.UniformFloatHyperparameter):
            value = float(trial.suggest_float(name=hp_name, low=hp.lower, high=hp.upper, log=hp.log))

        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            value = int(trial.suggest_int(name=hp_name, low=hp.lower, high=hp.upper, log=hp.log))

        elif isinstance(hp, CS.CategoricalHyperparameter):
            hp_type = type(hp.default_value)
            value = hp_type(trial.suggest_categorical(name=hp_name, choices=hp.choices))

        elif isinstance(hp, CS.OrdinalHyperparameter):
            num_vars = len(hp.sequence)
            index = trial.suggest_int(hp_name, low=0, high=num_vars - 1, log=False)
            hp_type = type(hp.default_value)
            value = hp.sequence[index]
            value = hp_type(value)

        else:
            raise ValueError(f'Please implement the support for hps of type {type(hp)}')

        config[hp.name] = value
    return config


def precompute_sh_iters(min_budget: Union[int, float], max_budget: Union[int, float], eta: Union[int, float]) -> int:
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    return max_SH_iter


def precompute_budgets(max_budget, eta, max_SH_iter):
    s0 = -np.linspace(start=max_SH_iter - 1,  stop=0, num=max_SH_iter)
    budgets = max_budget * np.power(eta, s0)
    return budgets


__all__ = [OptunaRandomSearchOptimizer,
           OptunaCMAESHyperBandOptimizer,
           OptunaTPEHyperbandOptimizer,
           OptunaTPEMedianStoppingOptimizer,
           sample_config_from_optuna]
