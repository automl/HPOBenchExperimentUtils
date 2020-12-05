import logging
from pathlib import Path
from typing import Union, Dict
import numpy as np
import time
import json

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
import HPOBenchExperimentUtils.utils.emukit_utils as emukit_utils
from HPOBenchExperimentUtils.utils.utils import get_mandatory_optimizer_setting
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

import ConfigSpace as cs
from emukit.core import ParameterSpace, InformationSourceParameter
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.optimization import MultiSourceAcquisitionOptimizer, GradientAcquisitionOptimizer
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO, _fit_gumbel
from emukit.core.acquisition import Acquisition
from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.model_wrappers import GPyMultiOutputWrapper
from GPy.kern import RBF

_log = logging.getLogger(__name__)


class GPwithMUMBO(SingleFidelityOptimizer):
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

        self.emukit_fidelity = InformationSourceParameter(self.info_sources.shape[0])
        self.fidelity_emukit_to_cs = lambda s: {self.main_fidelity.name: self.info_sources[s]}

        def wrapper(x: np.ndarray):
            """
            Remember that for MUMBO the search space is augmented with a discrete parameter for the fidelity index.
            This wrapper simply parses the configuration and fidelity and sets them up for calling the underlying
            benchmark objective function.
            """

            _log.debug("Benchmark wrapper received input %s." % str(x))
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            results = []
            for i in range(x.shape[0]):
                _log.debug("Extracted configuration: %s" % str(x[i, :-1]))
                _log.debug("Extracted fidelity value: %s" % str(self.info_sources[int(x[i, -1])]))
                fidelity = self.fidelity_emukit_to_cs(int(x[i, -1]))
                config = cs.Configuration(self.original_space,
                                          values={name: func(i) for (name, func), i in zip(self.to_cs, x[i, :-1])})
                res = benchmark.objective_function(config, fidelity=fidelity)
                _log.debug("Benchmark evaluation results: %s" % str(res))
                results.append(res["function_value"])
            results = np.asarray(results)
            return results if results.ndim == 2 else np.expand_dims(results, axis=1)

        self.benchmark_caller = wrapper

        # self.n_init = get_mandatory_optimizer_setting(settings, "num_init_evals")
        self.init_samples_per_dim = get_mandatory_optimizer_setting(settings, "init_samples_per_dim")
        # self.trajectory_samples_per_dim = get_mandatory_optimizer_setting(settings, "trajectory_samples_per_dim")
        self.gp_settings = {
            "n_optimization_restarts": get_mandatory_optimizer_setting(settings, "n_optimization_restarts")
        }
        self.mumbo_settings = {
            "num_mc_samples": get_mandatory_optimizer_setting(settings, "num_mc_samples"),
            "grid_size": get_mandatory_optimizer_setting(settings, "grid_size")
        }



    def _setup_model(self):

        augmented_space = ParameterSpace([*(self.emukit_space.parameters), self.emukit_fidelity])
        initial_design = RandomDesign(augmented_space)

        X_init = initial_design.get_samples(self.init_samples_per_dim * augmented_space.dimensionality)
        Y_init = np.asarray([self.benchmark_caller(X_init[i, :]) for i in range(X_init.shape[0])]).reshape(-1, 1)

        fidelity_kernels = []
        for _ in range(len(self.emukit_fidelity.bounds)):
            kernel = RBF(self.emukit_space.dimensionality)
            # TODO: Design decision. Do we care about these values?
            kernel.lengthscale.constrain_bounded(0.01, 0.5)
            fidelity_kernels.append(kernel)

        multi_fidelity_kernel = LinearMultiFidelityKernel(fidelity_kernels)
        n_fidelity_vals = self.info_sources.shape[0]
        gpy_model = GPyLinearMultiFidelityModel(X=X_init, Y=Y_init, kernel=multi_fidelity_kernel,
                                                n_fidelities=n_fidelity_vals)

        # TODO: Design decision. Do we care about this value?
        gpy_model.likelihood.Gaussian_noise.fix(0.1)
        for i in range(1, len(self.emukit_fidelity.bounds)):
            getattr(gpy_model.likelihood, f"Gaussian_noise_{i}").fix(0.1)

        model = GPyMultiOutputWrapper(gpy_model, n_outputs=2,
                                      n_optimization_restarts=self.gp_settings["n_optimization_restarts"],
                                      verbose_optimization=False)
        model.optimize()

        cost_acquisition = Cost(np.linspace(start=1. / n_fidelity_vals, stop=1.0, num=n_fidelity_vals))
        mumbo_acquisition = MUMBO(model, augmented_space, num_samples=self.mumbo_settings["num_mc_samples"],
                                  grid_size=self.mumbo_settings["grid_size"]) / cost_acquisition
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(augmented_space),
                                                                space=augmented_space)

        self.optimizer = BayesianOptimizationLoop(space=augmented_space, model=model, acquisition=mumbo_acquisition,
                                                  update_interval=1, batch_size=1,
                                                  acquisition_optimizer=acquisition_optimizer)
        self.optimizer.loop_start_event.append(emukit_utils.get_init_trajectory_hook(self.output_dir))
        self.optimizer.iteration_end_event.append(emukit_utils.get_trajectory_hook(self.output_dir))

    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting GP optimizer with MUMBO acquisition.")
        self._setup_model()
        self.optimizer.run_loop(user_function=self.benchmark_caller,
                                stopping_condition=emukit_utils.InfiniteStoppingCondition())
        _log.info("GP optimizer with MUMBO acquisition finished.")
        return self.output_dir


# Define cost of different fidelities as acquisition function
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)
