import logging
from pathlib import Path
from typing import Union, Dict
import sys
import numpy as np

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.utils.utils import get_mandatory_optimizer_setting
import HPOBenchExperimentUtils.utils.emukit_utils as emukit_utils
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

import ConfigSpace as cs

from emukit.examples.fabolas import fmin_fabolas, FabolasModel
from emukit.core import ParameterSpace
from emukit.core.loop import UserFunctionWrapper
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.optimization import MultiSourceAcquisitionOptimizer, GradientAcquisitionOptimizer
from emukit.core.acquisition import IntegratedHyperParameterAcquisition, acquisition_per_expected_cost
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import \
    CostSensitiveBayesianOptimizationLoop

_log = logging.getLogger(__name__)

class FabolasOptimizer(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        super().__init__(benchmark, settings, output_dir, rng)
        self.original_space = self.benchmark.get_configuration_space()
        self.emukit_space, self.to_emu, self.to_cs = emukit_utils.generate_space_mappings(self.original_space)
        if isinstance(self.main_fidelity, cs.UniformIntegerHyperparameter):
            _log.debug("Treating integer fidelity parameter %s as the main fidelity used for dataset subsampling." %
                       self.main_fidelity.name)
            self.s_min = max(self.main_fidelity.lower, 1)
            self.s_max = self.main_fidelity.upper
            self.subsample_to_cs_fidel = lambda x: x
        elif isinstance(self.main_fidelity, cs.UniformFloatHyperparameter):
            _log.debug("Treating float fidelity parameter %s as the main fidelity used for dataset subsampling." %
                       self.main_fidelity.name)
            assert hasattr(benchmark.benchmark, 'X_train'), "The benchmark object is expected to have an attribute " \
                                                            "'X_train' in order to be compatible with FABOLAS."
            assert hasattr(benchmark.benchmark, 'y_train'), "The benchmark object is expected to have an attribute " \
                                                            "'y_train' in order to be compatible with FABOLAS."
            assert 0.0 <= self.main_fidelity.lower and self.main_fidelity.upper <= 1.0
            self.s_min = max(self.main_fidelity.lower * benchmark.benchmark.y_train.shape[0], 1)
            self.s_max = self.main_fidelity.upper * benchmark.benchmark.y_train.shape[0]
            self.subsample_to_cs_fidel = lambda x: x / self.s_max
        else:
            raise RuntimeError("The benchmark's main fidelity parameter must be either a float or int, found "
                               "type %s" % type(self.main_fidelity))

        def wrapper(x, s):
            _log.debug("Calling objective function with configuration %s and dataset size %.2f/%.2f." %
                       (x, s, self.s_max))
            x = cs.Configuration(self.original_space, values={name: func(i) for (name, func), i in zip(self.to_cs, x)})
            res = benchmark.objective_function(x, fidelity={self.main_fidelity.name: self.subsample_to_cs_fidel(s)})
            return res["function_value"], res["cost"]

        self.benchmark_caller = wrapper
        self.n_init = get_mandatory_optimizer_setting(settings, "num_init_evals")

    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting FABOLAS optimizer.")
        res = fmin_fabolas(func=self.benchmark_caller, space=self.emukit_space, s_min=self.s_min, s_max=self.s_max,
                           n_iters=sys.maxsize, n_init=self.n_init,
                           marginalize_hypers=self.settings["marginalize_hypers"])
        _log.info("FABOLAS optimizer finished.")
        return self.output_dir


class FabolasWithMUMBO(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        super().__init__(benchmark, settings, output_dir, rng)
        self.original_space = self.benchmark.get_configuration_space()
        self.emukit_space, self.to_emu, self.to_cs = emukit_utils.generate_space_mappings(self.original_space)
        if isinstance(self.main_fidelity, cs.UniformFloatHyperparameter):
            num_fidelity_values = get_mandatory_optimizer_setting(
                settings, "num_fidelity_values", err_msg="When using a continuous fidelity parameter, number of "
                                                         "discrete fidelity levels must be specified in the parameter "
                                                         "'num_fidelity_values'")
            _log.debug("Discretizing the main fidelity %s for use with MUMBO into %d fidelity levels." %
                       (self.main_fidelity.name, num_fidelity_values))
            self.info_sources = np.linspace(self.min_budget, self.max_budget, num_fidelity_values)

        elif isinstance(self.main_fidelity, cs.OrdinalHyperparameter):
            self.info_sources = np.asarray(self.main_fidelity.get_seq_order())
        elif isinstance(self.main_fidelity, cs.CategoricalHyperparameter):
            self.info_sources = np.asarray(self.main_fidelity.choices)
        elif isinstance(self.main_fidelity, cs.UniformIntegerHyperparameter):
            self.info_sources = np.arange(start=self.min_budget, stop=self.max_budget + 1)

        self.emukit_fidelity = emukit_utils.SmarterInformationSourceParameter(self.info_sources.shape[0], start_ind=1)
        self.fidelity_emukit_to_cs = lambda s: {self.main_fidelity.name: self.info_sources[int(s)-1]}

        def wrapper(inp):
            _log.debug("Benchmark wrapper received input %s." % str(inp))
            if inp.ndim == 1:
                inp = np.expand_dims(inp, axis=0)

            yvals, costs = [], []
            for i in range(inp.shape[0]):
                x, s = inp[0, :-1], inp[0, -1]
                _log.debug("Calling objective function with configuration %s and fidelity index %s." % (x, s))
                config = cs.Configuration(self.original_space, values={name: func(i) for (name, func), i in zip(self.to_cs, x)})
                fidelity = self.fidelity_emukit_to_cs(s)
                _log.debug("Generated configuration %s, fidelity %s" % (config, fidelity))
                res = benchmark.objective_function(config, fidelity=fidelity)
                y, c = res["function_value"], res["cost"]
                yvals.append(y)
                costs.append(c)

            return np.asarray(yvals).reshape(-1, 1), np.asarray(costs).reshape(-1, 1)

        self.benchmark_caller = wrapper
        self.n_init = get_mandatory_optimizer_setting(settings, "num_init_evals")

        self.optimizer_settings = {
            "update_interval": get_mandatory_optimizer_setting(settings, "update_interval"),
            "marginalize_hypers": get_mandatory_optimizer_setting(settings, "marginalize_hypers"),
        }

        self.mumbo_settings = {
            "num_mc_samples": get_mandatory_optimizer_setting(settings, "num_mc_samples"),
            "grid_size": get_mandatory_optimizer_setting(settings, "grid_size")
        }

        _log.info("Finished reading all settings for FABOLAS optimizer with MUMBO acquisition.")

    def _setup_model(self):
        initial_design = LatinDesign(self.emukit_space)
        s_min, s_max = self.emukit_fidelity.bounds[0]

        grid = initial_design.get_samples(self.n_init)
        n_reps = self.n_init // self.info_sources.shape[0] + 1
        sample_fidelities = np.expand_dims(np.tile(np.arange(s_min, s_max), n_reps)[:self.n_init], 1)
        X_init = np.concatenate((grid, sample_fidelities), axis=1)
        res = np.array(list(map(self.benchmark_caller, X_init))).reshape((-1, 2))
        Y_init = res[:, 0][:, None]
        cost_init = res[:, 1][:, None]
        _log.debug("Generated %d warm-start samples." % X_init.shape[0])

        extended_space = ParameterSpace([*(self.emukit_space.parameters), self.emukit_fidelity])

        model_objective = FabolasModel(X_init=X_init, Y_init=Y_init, s_min=s_min, s_max=s_max)
        model_cost = FabolasModel(X_init=X_init, Y_init=cost_init, s_min=s_min, s_max=s_max)
        _log.debug("Initialized objective and cost estimation models")

        if self.optimizer_settings["marginalize_hypers"]:
            acquisition_generator = lambda model: MUMBO(
                model=model_objective, space=extended_space, target_information_source_index=s_max,
                num_samples=self.mumbo_settings["num_mc_samples"], grid_size=self.mumbo_settings["grid_size"])

            entropy_search = IntegratedHyperParameterAcquisition(model_objective, acquisition_generator)
        else:
            entropy_search = MUMBO(
                model=model_objective, space=extended_space, target_information_source_index=s_max,
                num_samples=self.mumbo_settings["num_mc_samples"], grid_size=self.mumbo_settings["grid_size"])

        acquisition = acquisition_per_expected_cost(entropy_search, model_cost)
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(extended_space),
                                                                space=extended_space)
        _log.debug("MUMBO acquisition function ready.")

        # TODO: Insert note in documentation, hold discussion over change of acquisition optimizer from RandomSearch
        # acquisition_optimizer = RandomSearchAcquisitionOptimizer(
        #     extended_space, num_eval_points=self.optimizer_settings["num_eval_points"])

        self.optimizer = CostSensitiveBayesianOptimizationLoop(
            space=extended_space, model_objective=model_objective, model_cost=model_cost, acquisition=acquisition,
            update_interval=self.optimizer_settings["update_interval"], acquisition_optimizer=acquisition_optimizer)

        self.optimizer.loop_start_event.append(emukit_utils.get_init_trajectory_hook(self.output_dir))
        self.optimizer.iteration_end_event.append(emukit_utils.get_trajectory_hook(self.output_dir))
        _log.info("FABOLAS optimizer with MUMBO acquisition initialized and ready to run.")

    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting FABOLAS optimizer with MUMBO acquisition function.")
        self._setup_model()
        self.optimizer.run_loop(UserFunctionWrapper(self.benchmark_caller, extra_output_names=["cost"]),
                                emukit_utils.InfiniteStoppingCondition())
        _log.info("FABOLAS optimizer finished.")
        return self.output_dir
