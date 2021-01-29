import logging
from pathlib import Path
from typing import Union, Dict, Tuple, Sequence
import sys
import numpy as np
from math import log2
import enum

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.utils.utils import get_mandatory_optimizer_setting, standard_rng_init
import HPOBenchExperimentUtils.utils.emukit_utils as emukit_utils
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

import ConfigSpace as cs

from emukit.examples.fabolas import fmin_fabolas, FabolasModel
from emukit.examples.fabolas.continuous_fidelity_entropy_search import ContinuousFidelityEntropySearch
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop import UserFunctionWrapper
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.initial_designs import RandomDesign
from emukit.core.optimization import MultiSourceAcquisitionOptimizer, GradientAcquisitionOptimizer, \
    RandomSearchAcquisitionOptimizer
from emukit.core.acquisition import IntegratedHyperParameterAcquisition, acquisition_per_expected_cost
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import \
    CostSensitiveBayesianOptimizationLoop

_log = logging.getLogger(__name__)


class AcquisitionTypes(enum.Enum):
    MTBO = "mtbo"
    MUMBO = "mumbo"

initial_designs = {
    "random": RandomDesign,
    "latin": LatinDesign
}

_fidelity_parameter_names = ["subsample", "dataset_fraction"]

# noinspection PyPep8Naming
class FabolasOptimizer(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        super().__init__(benchmark, settings, output_dir, rng)

        # The benchmark defined its configuration space as a ConfigSpace.ConfigurationSpace object. This must be parsed
        # into emukit's version, an emukit.ParameterSpace object. Additionally, we need mappings between configurations
        # defined in either of the two conventions.
        self.original_space = self.benchmark.get_configuration_space()
        self.optimizer_settings = {
            "update_interval": get_mandatory_optimizer_setting(settings, "update_interval"),
            "marginalize_hypers": get_mandatory_optimizer_setting(settings, "marginalize_hypers"),
            "initial_design": str(get_mandatory_optimizer_setting(settings, "initial_design")).lower()
        }

        acquisition_type = str(get_mandatory_optimizer_setting(settings, "acquisition")).lower()
        if acquisition_type == AcquisitionTypes.MUMBO.value:
            # Fabolas optimizer with the default acquisition function replaced with MTBO MUMBO acquisition.
            # cf. Section 4.3 of the MUMBO paper: https://arxiv.org/pdf/2006.12093.pdf
            self.acquisition_type = AcquisitionTypes.MUMBO
            self.mumbo_settings = {
                "num_mc_samples": get_mandatory_optimizer_setting(settings, "num_mc_samples"),
                "grid_size": get_mandatory_optimizer_setting(settings, "grid_size")
            }
        elif acquisition_type == AcquisitionTypes.MTBO.value:
            # Fabolas optimizer with the default MTBO acquisition function.
            self.acquisition_type = AcquisitionTypes.MTBO
            self.mtbo_settings = {
                "num_eval_points": get_mandatory_optimizer_setting(settings, "num_eval_points")
            }
        else:
            raise ValueError("Fabolas optimizer does not recognize acquisition function %s. Expected either 'mumbo' or "
                             "'mtbo'." % str(self.acquisition_type))

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
                x, s = inp[i, :-1], inp[i, -1]
                _log.debug("Calling objective function with configuration %s and fidelity index %s." % (x, s))
                config = cs.Configuration(self.original_space, values=self.to_cs(x))
                fidelity = self.fidelity_emukit_to_cs(s)
                _log.debug("Generated configuration %s, fidelity %s" % (config, fidelity))
                res = benchmark.objective_function(config, fidelity=fidelity,
                                                   **self.settings_for_sending)
                y, c = res["function_value"], res["cost"]
                yvals.append(y)
                costs.append(c)

            return np.asarray(yvals).reshape(-1, 1), np.asarray(costs).reshape(-1, 1)

        self.benchmark_caller = wrapper
        self.n_init = int(get_mandatory_optimizer_setting(settings, "init_samples_per_dim") *
                          self.emukit_space.dimensionality)

        _log.info("Finished reading all settings for FABOLAS optimizer with MUMBO acquisition.")

    def _setup_fabolas_fidelity(self):
        # FABOLAS will only work on a fidelity parameter named in '_fidelity_parameter_names', with values in
        # (0.0, 1.0].
        if self.main_fidelity.name not in _fidelity_parameter_names:
            raise RuntimeError("Cannot process unrecognized fidelity parameter %s. Must be one of %s." %
                               (self.main_fidelity.name, str(_fidelity_parameter_names)))

        # As per the original sample code, the fidelity values were first discretized by FABOLAS, effectively running
        # continuous fidelity BO on top of Multi-Task BO, cf. emukit.examples.fabolas.fmin_fabolas.fmin().
        _log.debug("Discretizing the dataset sizes for use with FABOLAS into %d fidelity levels on a log2 scale." %
                   self.num_fidelity_values)

        if type(self.main_fidelity) == cs.OrdinalHyperparameter:
            # Assume that the OrdinalHyperparameter sequence contains exact budget sizes.
            self.main_fidelity: cs.OrdinalHyperparameter
            self.budgets = np.asarray(self.main_fidelity.sequence)
            s_min = self.budgets[0]
            s_max = self.budgets[-1]
            ordinal = True
        else:
            # Assume that the sample budgets need to be extracted from a range of values.
            # FABOLAS expects to sample the fidelity values on a log scale. The following code effectively ignores
            # the log attribute of the parameter.
            # As per the interface of FabolasModel, s_min and s_max should not be fractions but dataset sizes.
            s_min = max(self.min_budget * self.dataset_size, 1)
            s_max = max(self.max_budget * self.dataset_size, 1)
            self.budgets = np.rint(np.clip(np.power(0.5, np.arange(self.num_fidelity_values - 1, -1, -1)) * s_max,
                                       s_min, s_max)).astype(int)
            # Needed to avoid introducing more ifs.
            # def fid_op(s): return np.clip(s, s_min, s_max) / s_max
            ordinal = False

        if self.acquisition_type is AcquisitionTypes.MUMBO:
            # To summarize, MUMBO will be given the fidelity values as a list of indices, each corresponding to an
            # actual dataset_fraction value sampled on a log2 scale and stored in 'budgets'. Thus, the value retrieved
            # from budgets needs to be clipped to account for any numerical inconsistencies arising out of the chained
            # log and exp operations.
            if ordinal:
                # s is an index, budget contains integers, the fidelity is expected to be an exact integer
                def map_fn(s: int):
                    return {self.main_fidelity.name: self.budgets[int(s)]}
            else:
                # s is an index, budget contains integers, the fidelity is expected to be a fraction in [0.0, 1.0]
                def map_fn(s: int): return {self.main_fidelity.name: np.clip(self.budgets[int(s)],
                                                                             s_min, s_max) / s_max}

            self.fidelity_emukit_to_cs = map_fn

            # It was necessary to define a custom InformationSourceParameter here because of some minor issues that
            # FABOLAS had with a DiscreteParameter (the parent class of InformationSourceParameter) beginning at index
            # 0. Using a sub-class of InformationSourceParameter was, in turn, necessary, because there are internal
            # checks in place in MUMBO for that.
            # self.fabolas_fidelity = emukit_utils.SmarterInformationSourceParameter(self.budgets.shape[0], start_ind=1)
            self.fabolas_fidelity = InformationSourceParameter(self.budgets.shape[0])
        elif self.acquisition_type is AcquisitionTypes.MTBO:
            if ordinal:
                # s is a size in integers, budget contains integers, fidelity is expected to be an integer - find
                # closest fit for s in budgets
                def map_fn(s: int): return {self.main_fidelity.name: self.budgets[np.abs(self.budgets - s).argmin()]}
            else:
                # s is a size in integers, budget contains sizes in integers, fidelity is expected to be a fraction in
                # the range [0.0, 1.0]
                def map_fn(s: int): return {self.main_fidelity.name: np.clip(s, s_min, s_max) / s_max}

            self.fidelity_emukit_to_cs = map_fn
            self.fabolas_fidelity = ContinuousParameter("s", s_min, s_max)
        else:
            raise RuntimeError("Unexpected acquisition type %s" % self.acquisition_type)

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
        s_min, s_max = self.fabolas_fidelity.bounds[0]
        if self.acquisition_type is AcquisitionTypes.MUMBO:
            # For MUMBO, we sample fidelities as indices instead of actual dataset sizes
            sample_fidelities = np.expand_dims(np.tile(np.arange(s_min, s_max+1), n_reps)[:self.n_init], 1)
        else:
            # For MTBO, we directly sample the dataset sizes
            sample_fidelities = np.expand_dims(np.tile(self.budgets, n_reps)[:self.n_init], 1)

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

        extended_space = ParameterSpace([*self.emukit_space.parameters, self.fabolas_fidelity])

        if self.acquisition_type is AcquisitionTypes.MUMBO:
            # Insert MUMBO acquisition instead of FABOLAS' MTBO acquisition

            # The actual FABOLAS model comes into play here. Same as the FABOLAS example code. Note that here, we pass
            # the actual minimum and maximum budget/dataset size values instead of indices to FabolasModel. This is the
            # reason why the wrapper was needed.
            model_objective = FabolasModelMumboWrapper(budgets=self.budgets, X_init=X_init, Y_init=Y_init,
                                                       s_min=self.budgets[0], s_max=self.budgets[-1])
            model_cost = FabolasModelMumboWrapper(budgets=self.budgets, X_init=X_init, Y_init=cost_init,
                                                  s_min=self.budgets[0], s_max=self.budgets[-1])
            _log.debug("Initialized objective and cost estimation models")

            # ---------------------- ---------------------- ---------------------- ------------------ ---------------- #
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

            acquisition = acquisition_per_expected_cost(entropy_search, model_cost)
            # This was used in the MUMBO example code
            acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(extended_space),
                                                                    space=extended_space)
            _log.debug("MUMBO acquisition function ready.")
            # --------------------- --------------------- --------------------- --------------------- ---------------- #

            # Define the properties of the BO loop within which the chosen surrogate model (FABOLAS) and acquisition
            # function (MUMBO) are used for performing BO. Same as the FABOLAS example code.
            self.optimizer = CostSensitiveBayesianOptimizationLoop(
                space=extended_space, model_objective=model_objective, model_cost=model_cost, acquisition=acquisition,
                update_interval=self.optimizer_settings["update_interval"], acquisition_optimizer=acquisition_optimizer)

            # These are hooks that help us record the trajectory for an information theoretic acquisition function,
            # which cannot be handled otherwise by the Bookkeeper.
            self.optimizer.loop_start_event.append(emukit_utils.get_init_trajectory_hook(self.output_dir))
            self.optimizer.iteration_end_event.append(emukit_utils.get_trajectory_hook(self.output_dir, self.to_cs))
            _log.info("FABOLAS optimizer with MUMBO acquisition initialized and ready to run.")
        else:
            # AcquisitionType.MTBO
            # The default MTBO acquisition for FABOLAS.

            model_objective = FabolasModel(X_init=X_init, Y_init=Y_init, s_min=self.budgets[0], s_max=self.budgets[-1])
            model_cost = FabolasModel(X_init=X_init, Y_init=cost_init, s_min=self.budgets[0], s_max=self.budgets[-1])

            if self.optimizer_settings["marginalize_hypers"]:
                acquisition_generator = lambda model: ContinuousFidelityEntropySearch(
                    model=model_objective, space=extended_space,
                    target_fidelity_index=len(extended_space.parameters) - 1)
                entropy_search = IntegratedHyperParameterAcquisition(model_objective, acquisition_generator)
            else:
                entropy_search = ContinuousFidelityEntropySearch(
                    model=model_objective, space=extended_space,
                    target_fidelity_index=len(extended_space.parameters) - 1)

            acquisition = acquisition_per_expected_cost(entropy_search, model_cost)
            acquisition_optimizer = RandomSearchAcquisitionOptimizer(
                extended_space, num_eval_points=self.mtbo_settings["num_eval_points"])
            _log.debug("MTBO acquisition function ready.")
            # --------------------- --------------------- --------------------- --------------------- ---------------- #

            # Define the properties of the BO loop within which the chosen surrogate model (FABOLAS) and acquisition
            # function (MUMBO) are used for performing BO. Same as the FABOLAS example code.
            self.optimizer = CostSensitiveBayesianOptimizationLoop(
                space=extended_space, model_objective=model_objective, model_cost=model_cost, acquisition=acquisition,
                update_interval=self.optimizer_settings["update_interval"], acquisition_optimizer=acquisition_optimizer)

            _log.info("FABOLAS optimizer with MTBO acquisition initialized and ready to run.")

    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting FABOLAS optimizer.")

        # emukit does not expose any interface for setting a random seed any other way, so we reset the global seed here
        # Generating a new random number from the seed ensures that, for compatible versions of the numpy.random module,
        # the seeds remain predictable while still handling seed=None in a consistent manner.
        np.random.seed(standard_rng_init(self.rng).randint(0, 1_000_000))
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

    def __init__(self, budgets: Sequence[int], X_init: np.ndarray, **kwargs):
        # Initialize an actual FabolasModel instance
        self.budgets = np.asarray(budgets)
        super(FabolasModelMumboWrapper, self).__init__(self._idx_to_budgets(X_init), **kwargs)

    @property
    def X(self):
        return self._budgets_to_idx(super(FabolasModelMumboWrapper, self).X)

    def _budgets_to_idx(self, X):
        """ Given inputs that contain values from self.budgets as the fidelity values (last column), returns the
        same inputs with the values replaced by corresponding indices from self.budgets. """
        X_ = np.array(X, copy=True)
        X_[:, -1] = np.clip(np.searchsorted(self.budgets, X_[:, -1]), a_min=0, a_max=self.budgets.shape[0] - 1)
        return X_

    def _idx_to_budgets(self, X):
        """ Given inputs that contain array indices of self.budgets as the fidelity values (last column), returns the
        same inputs with the indices replaced by corresponding values from self.budgets. """
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
