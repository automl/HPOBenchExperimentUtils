import logging
from pathlib import Path
from typing import Union, Dict, Tuple, Sequence
import sys
import numpy as np
from math import log2

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.utils.utils import get_mandatory_optimizer_setting
import HPOBenchExperimentUtils.utils.emukit_utils as emukit_utils
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

import ConfigSpace as cs

from emukit.examples.fabolas import fmin_fabolas, FabolasModel
from emukit.core import ParameterSpace, InformationSourceParameter
from emukit.core.loop import UserFunctionWrapper
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.initial_designs import RandomDesign
from emukit.core.optimization import MultiSourceAcquisitionOptimizer, GradientAcquisitionOptimizer
from emukit.core.acquisition import IntegratedHyperParameterAcquisition, acquisition_per_expected_cost
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import \
    CostSensitiveBayesianOptimizationLoop

_log = logging.getLogger(__name__)

initial_designs = {
    "random": RandomDesign,
    "latin": LatinDesign
}


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
        self.n_init = int(get_mandatory_optimizer_setting(settings, "init_samples_per_dim") *
                          self.emukit_space.dimensionality)

    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting FABOLAS optimizer.")
        _ = fmin_fabolas(func=self.benchmark_caller, space=self.emukit_space, s_min=self.s_min, s_max=self.s_max,
                         n_iters=sys.maxsize, n_init=self.n_init,
                         marginalize_hypers=self.settings["marginalize_hypers"])
        _log.info("FABOLAS optimizer finished.")
        return self.output_dir


# Fabolas optimizer with the default acquisition function replaced with MTBO MUMBO acquisition.
# Ref. Section 4.3 of the MUMBO paper: https://arxiv.org/pdf/2006.12093.pdf
# noinspection PyPep8Naming
class FabolasWithMUMBO(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        super().__init__(benchmark, settings, output_dir, rng)

        # The benchmark defined its configuration space as a ConfigSpace.ConfigurationSpace object. This must be parsed
        # into emukit's version, an emukit.ParameterSpace object. Additionally, we need mappings between configurations
        # defined in either of the two conventions.
        self.original_space = self.benchmark.get_configuration_space()
        self.emukit_space, self.to_emu, self.to_cs = emukit_utils.generate_space_mappings(self.original_space)

        self.num_fidelity_values = get_mandatory_optimizer_setting(
            settings, "num_fidelity_values", err_msg="Number of discrete fidelity levels must be specified in the "
                                                     "parameter 'num_fidelity_values'. This defines the number of "
                                                     "individual dataset sizes that will be used by FABOLAS.")
        self.dataset_size = benchmark.benchmark.y_train.shape[0]
        self._setup_fabolas_fidelity()

        def wrapper(inp):
            """ Emukit requires this function to accept 2D inputs, with individual configurations aligned along axis 0
            and the various components of each configuration along axis 1. FABOLAS itself will only query one
            configuration at a time, but the interface must support multiple. """

            nonlocal self
            _log.debug("Benchmark wrapper received input %s." % str(inp))
            if inp.ndim == 1:
                inp = np.expand_dims(inp, axis=0)

            yvals, costs = [], []
            for i in range(inp.shape[0]):
                x, s = inp[0, :-1], inp[0, -1]
                _log.debug("Calling objective function with configuration %s and fidelity index %s." % (x, s))
                config = cs.Configuration(self.original_space,
                                          values={name: func(i) for (name, func), i in zip(self.to_cs, x)})
                fidelity = self.fidelity_emukit_to_cs(s)
                _log.debug("Generated configuration %s, fidelity %s" % (config, fidelity))
                res = benchmark.objective_function(config, fidelity=fidelity)
                y, c = res["function_value"], res["cost"]
                yvals.append(y)
                costs.append(c)

            return np.asarray(yvals).reshape(-1, 1), np.asarray(costs).reshape(-1, 1)

        self.benchmark_caller = wrapper
        self.n_init = int(get_mandatory_optimizer_setting(settings, "init_samples_per_dim") *
                          self.emukit_space.dimensionality)

        self.optimizer_settings = {
            "update_interval": get_mandatory_optimizer_setting(settings, "update_interval"),
            "marginalize_hypers": get_mandatory_optimizer_setting(settings, "marginalize_hypers"),
            "initial_design": str(get_mandatory_optimizer_setting(settings, "initial_design")).lower()
        }

        self.mumbo_settings = {
            "num_mc_samples": get_mandatory_optimizer_setting(settings, "num_mc_samples"),
            "grid_size": get_mandatory_optimizer_setting(settings, "grid_size")
        }

        _log.info("Finished reading all settings for FABOLAS optimizer with MUMBO acquisition.")

    def _setup_fabolas_fidelity(self):
        # FABOLAS will only work on a fidelity named 'dataset_fraction', with values in (0.0, 1.0].
        if self.main_fidelity.name != "dataset_fraction":
            raise RuntimeError("Cannot process unrecognized fidelity parameter %s." % self.main_fidelity.name)
        assert isinstance(self.main_fidelity, cs.UniformFloatHyperparameter), \
            "The fidelity parameter 'dataset_fraction' should be of type %s, found %s." % \
            (str(cs.UniformFloatHyperparameter.__class__), str(type(self.main_fidelity)))

        # As per the original sample code, the fidelity values were first discretized by FABOLAS, effectively running
        # continuous fidelity BO on top of Multi-Task BO, cf. emukit.examples.fabolas.fmin_fabolas.fmin().
        _log.debug("Discretizing the dataset sizes for use with FABOLAS into %d fidelity levels on a log2 scale." %
                   self.num_fidelity_values)

        # FABOLAS expects to sample the fidelity values on a log scale. The following code effectively ignores the log
        # attribute of the parameter.
        # As per the interface of FabolasModel, s_min and s_max should not be fractions but dataset sizes.
        s_min = max(self.min_budget * self.dataset_size, 1)
        s_max = max(self.max_budget * self.dataset_size, 1)
        self.budgets = np.rint(np.logspace(start=log2(s_min), stop=log2(s_max), num=self.num_fidelity_values, base=2.0))

        # It was necessary to define a custom InformationSourceParameter here because of some minor issues that FABOLAS
        # had with a DiscreteParameter (the parent class of InformationSourceParameter) beginning at index 0. Using
        # a sub-class of InformationSourceParameter was, in turn, necessary, because there are internal checks in place
        # in MUMBO for that.
        # self.emukit_fidelity = emukit_utils.SmarterInformationSourceParameter(self.budgets.shape[0], start_ind=1)
        self.emukit_fidelity = InformationSourceParameter(self.budgets.shape[0])

        # To summarize, FABOLAS will be given the fidelity values as a list of indices, each corresponding to an actual
        # dataset_fraction value sampled on a log2 scale and stored in 'budgets'. Thus, the value retrieved from
        # budgets needs to be clipped to account for any numerical inconsistencies arising out of the chained log
        # and exp operations.
        self.fidelity_emukit_to_cs = lambda s: {
            self.main_fidelity.name: min(self.max_budget, max(self.min_budget, self.budgets[int(s) - 1]))
        }

    def _setup_model(self):
        """
        This is almost entirely boilerplate code required to setup a model to work under the emukit framework. This
        code has been adapted from the convenience wrappers fmin.fmin_fabolas() and fabolas_loop.FabolasLoop from the
        FABOLAS example in emukit.examples.fabolas, here:
        https://github.com/EmuKit/emukit/tree/96299e99c5c406b46baf6f0f0bbea70950566918/emukit/examples/fabolas
        """

        # ############################################################################################################ #
        # Ref: emukit.examples.fabolas.fmin.fmin_fabolas()
        # https://github.com/EmuKit/emukit/blob/96299e99c5c406b46baf6f0f0bbea70950566918/emukit/examples/fabolas/fmin.py

        # Generate warm-start samples. Using the implementation provided in the FABOLAS example, but B.2 in the
        # appendix of the MUMBO paper indicates RandomDesign should be used here instead. Therefore, exposed as a
        # hyperparameter.
        initial_design = initial_designs[self.optimizer_settings["initial_design"]](self.emukit_space)
        grid = initial_design.get_samples(self.n_init)
        n_reps = self.n_init // self.budgets.shape[0] + 1

        # Samples for the fidelity values. Same as the FABOLAS example code.
        s_min, s_max = self.emukit_fidelity.bounds[0]
        sample_fidelities = np.expand_dims(np.tile(np.arange(s_min, s_max+1), n_reps)[:self.n_init], 1)

        # Append sampled fidelity values to sampled configurations and perform evaluations. Same as the FABOLAS example
        # code.
        X_init = np.concatenate((grid, sample_fidelities), axis=1)
        res = np.array(list(map(self.benchmark_caller, X_init))).reshape((-1, 2))
        Y_init = res[:, 0][:, None]
        cost_init = res[:, 1][:, None]
        _log.debug("Generated %d warm-start samples." % X_init.shape[0])
        # ############################################################################################################ #

        # ############################################################################################################ #
        # Ref: emukit.examples.fabolas.fabolas_loop.FabolasLoop
        # https://github.com/EmuKit/emukit/blob/96299e99c5c406b46baf6f0f0bbea70950566918/emukit/examples/fabolas/fabolas_loop.py

        extended_space = ParameterSpace([*self.emukit_space.parameters, self.emukit_fidelity])

        # The actual FABOLAS model comes into play here. Same as the FABOLAS example code. Note that here, we pass the
        # actual minimum and maximum budget/dataset size values instead of indices to FabolasModel. This is the reason
        # why the wrapper was needed.
        model_objective = FabolasModelMumboWrapper(budgets=self.budgets, X_init=X_init, Y_init=Y_init,
                                                   s_min=self.budgets[0], s_max=self.budgets[-1])
        model_cost = FabolasModelMumboWrapper(budgets=self.budgets, X_init=X_init, Y_init=cost_init,
                                              s_min=self.budgets[0], s_max=self.budgets[-1])
        _log.debug("Initialized objective and cost estimation models")

        # ---------------------- ---------------------- ---------------------- ---------------------- ---------------- #
        # Insert MUMBO acquisition instead of FABOLAS' MTBO acquisition
        # Ref. Section 4.3 of the MUMBO paper: https://arxiv.org/pdf/2006.12093.pdf
        if self.optimizer_settings["marginalize_hypers"]:
            acquisition_generator = lambda model: MUMBO(
                model=model_objective, space=extended_space, target_information_source_index=s_max,
                num_samples=self.mumbo_settings["num_mc_samples"], grid_size=self.mumbo_settings["grid_size"])

            entropy_search = IntegratedHyperParameterAcquisition(model_objective, acquisition_generator)
        else:
            entropy_search = MUMBO(
                model=model_objective, space=extended_space, target_information_source_index=s_max,
                num_samples=self.mumbo_settings["num_mc_samples"], grid_size=self.mumbo_settings["grid_size"])

        # TODO: Insert note in documentation, hold discussion over change of acquisition optimizer from RandomSearch
        acquisition = acquisition_per_expected_cost(entropy_search, model_cost)
        # This was used in the MUMBO example code
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(extended_space),
                                                                space=extended_space)
        # Whereas this was used in the original FABOLAS code
        # acquisition_optimizer = RandomSearchAcquisitionOptimizer(
        #     extended_space, num_eval_points=self.optimizer_settings["num_eval_points"])
        _log.debug("MUMBO acquisition function ready.")
        # ---------------------- ---------------------- ---------------------- ---------------------- ---------------- #

        # Define the properties of the BO loop within which the chosen surrogate model (FABOLAS) and acquisition
        # function (MUMBO) are used for performing BO. Same as the FABOLAS example code.
        self.optimizer = CostSensitiveBayesianOptimizationLoop(
            space=extended_space, model_objective=model_objective, model_cost=model_cost, acquisition=acquisition,
            update_interval=self.optimizer_settings["update_interval"], acquisition_optimizer=acquisition_optimizer)
        # ############################################################################################################ #

        # These are hooks that help us record the trajectory for an information theoretic acquisition function, which
        # cannot be handled otherwise by the Bookkeeper.
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


# noinspection PyPep8Naming
class FabolasModelMumboWrapper(FabolasModel):
    """ A wrapper that allows MUMBO to properly interface with FabolasModel instances on account of their different
    treatments of the fidelity parameter. Essentially, a MUMBO acquisition will always call the predict() method
    while passing the fidelity value as an index in [0, N-1] for N fidelity sources and expects the underlying model to
    appropriately handle the indices. FabolasModel, on the other hand, expects the predict() call to receive a dataset
    size integer as the last column of the input matrix. This gap is bridged using this wrapper. """

    def __init__(self, budgets: Sequence[int], **kwargs):
        # Initialize an actual FabolasModel instance
        super(FabolasModelMumboWrapper, self).__init__(**kwargs)
        self.budgets = np.asarray(budgets)

    @property
    def X(self):
        return self._budgets_to_idx(super(FabolasModelMumboWrapper, self).X)

    def _budgets_to_idx(self, X):
        X_ = np.array(X, copy=True)
        X_[:, -1] = np.clip(np.searchsorted(self.budgets, X_[:, -1]), a_min=0, a_max=self.budgets.shape[0] - 1)
        return X_

    def _idx_to_budgets(self, X):
        X_ = np.array(X, copy=True)
        X_[:, -1] = self.budgets[np.asarray(X[:, -1], int)]
        return X_

    def set_data(self, X, Y):
        return super(FabolasModelMumboWrapper, self).set_data(self._idx_to_budgets(X), Y)

    def predict(self, X):
        return super(FabolasModelMumboWrapper, self).predict(self._idx_to_budgets(X))

    def predict_covariance(self, X: np.ndarray, with_noise: bool = True) -> np.ndarray:
        return super(FabolasModelMumboWrapper, self).predict_covariance(self._idx_to_budgets(X), with_noise)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super(FabolasModelMumboWrapper, self).predict_with_full_covariance(self._idx_to_budgets(X))

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return super(FabolasModelMumboWrapper, self).get_covariance_between_points(
            self._idx_to_budgets(X1),
            self._idx_to_budgets(X2)
        )

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super(FabolasModelMumboWrapper, self).get_prediction_gradients(self._idx_to_budgets(X))

    def get_joint_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super(FabolasModelMumboWrapper, self).get_joint_prediction_gradients(self._idx_to_budgets(X))

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        return super(FabolasModelMumboWrapper, self).calculate_variance_reduction(
            self._idx_to_budgets(x_train_new),
            self._idx_to_budgets(x_test)
        )
