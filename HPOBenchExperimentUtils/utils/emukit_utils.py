import logging
from typing import Tuple, Callable, Sequence
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter, OneHotEncoding, CategoricalParameter, \
    Parameter
from emukit.core.loop import LoopState, OuterLoop, StoppingCondition
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
import ConfigSpace as cs
from math import log, exp
from pathlib import Path
import numpy as np
import time
import json

_log = logging.getLogger(__name__)

MUMBO_TRAJECTORY_FILE = "mumbo_trajectory.json"


class InfiniteStoppingCondition(StoppingCondition):
    """ Implements a simple infinite stopping condition. """
    def should_stop(self, loop_state: LoopState) -> bool:
        return False


def _handle_uniform_float(param: cs.UniformFloatHyperparameter) -> Tuple[ContinuousParameter, Callable, Callable]:
    """ Generate a mapping for a UniformFloatHyperparameter object. """
    min_val, max_val = param.lower, param.upper
    if param.log:
        min_val = log(min_val)
        max_val = log(max_val)
        def map_to_emu(x): return log(x)
        def map_to_cs(x): return np.clip(exp(x), param.lower, param.upper)
    else:
        def map_to_emu(x): return x
        def map_to_cs(x): return np.clip(x, param.lower, param.upper)
    emukit_param = ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)

    return emukit_param, map_to_emu, map_to_cs


def _handle_uniform_int(param: cs.UniformIntegerHyperparameter) -> Tuple[ContinuousParameter, Callable, Callable]:
    """ Generate a mapping for a UniformIntegerHyperparameter object. """

    if param.log:
        min_val, max_val = log(param.lower), log(param.upper)
        def map_to_emu(x): return log(x)
        def map_to_cs(x): return np.clip(np.rint(exp(x)).astype(int), param.lower, param.upper)
    else:
        min_val, max_val = param.lower, param.upper
        def map_to_emu(x): return x
        def map_to_cs(x): return np.rint(x).astype(int)

    emukit_param = ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)
    return emukit_param, map_to_emu, map_to_cs


def _handle_categorical(param: cs.CategoricalHyperparameter) -> Tuple[DiscreteParameter, Callable, Callable]:
    """ Generate a mapping for a CategoricalHyperparameter object. """

    values = param.choices
    encoding = OneHotEncoding(values)
    def map_to_emu(x): return encoding.get_encoding(x)
    def map_to_cs(x): return encoding.get_category(x)
    emukit_param = CategoricalParameter(param.name, encoding)
    return emukit_param, map_to_emu, map_to_cs


def _handle_ordinal(param: cs.OrdinalHyperparameter) -> Tuple[DiscreteParameter, Callable, Callable]:
    """ Generate a mapping for a CategoricalHyperparameter object. """

    values = param.sequence
    def map_to_emu(x): return values.index(x)
    def map_to_cs(x): return values[np.rint(x).astype(int).squeeze()]
    emukit_param = DiscreteParameter(name=param.name, domain=list(range(param.num_elements)))
    return emukit_param, map_to_emu, map_to_cs


param_map = {
    cs.UniformFloatHyperparameter: _handle_uniform_float,
    cs.UniformIntegerHyperparameter: _handle_uniform_int,
    cs.CategoricalHyperparameter: _handle_categorical,
    cs.OrdinalHyperparameter: _handle_ordinal
}


def generate_space_mappings(cspace: cs.ConfigurationSpace, valid_types: Sequence[str] = None) -> \
        Tuple[ParameterSpace, Sequence[Tuple[str, Callable]], Callable]:
    """
    Map a ConfigSpace.ConfigurationSpace object to an emukit compatible version and generate the relevant mappings
    to work across the two spaces. An optional list of strings defining valid ConfigSpace.Hyperparameter sub-classes
    can be passed as well. By default, all known sub-classes are accepted. Known sub-classes can be specified by the
    following strings:

    float - ConfigSpace.UniformFloatHyperparameter
    int - ConfigSpace.UniformIntegerHyperparameter
    categorical - ConfigSpace.CategoricalHyperparameter [always one-hot encoded]
    ordinal - ConfigSpace.OrdinalHyperparameter

    Returns space, to_emu, to_cs: space is an emukit.core.ParameterSpace object containing the emukit compatible
    equivalent configuration space, to_emu is a list of 2-tuples containing the parameter names and their respective
    parser functions for converting configurations sampled from ConfigSpace to emukit compatible values, and to_cs is
    a single parser function that converts a list of values corresponding to a single configuration parsed from 'space'
    into a dictionary of values that can be directly fed into ConfigSpace.Configuration along with the input 'cspace'
    to generate an appropriate ConfigSpace.Configuration object.
    """

    space = []
    to_emu = []
    to_cs = []
    all_known_types = {
        "float": cs.UniformFloatHyperparameter,
        "int": cs.UniformIntegerHyperparameter,
        "categorical": cs.CategoricalHyperparameter,
        "ordinal": cs.OrdinalHyperparameter
    }
    if valid_types is None:
        valid_types = all_known_types.values()
    else:
        valid_types = [all_known_types[k] for k in valid_types]
    _log.debug(f"Generate mappings for ConfigurationSpace object, accepting parameters of types: {valid_types}")

    for parameter in cspace.get_hyperparameters():
        ptype = type(parameter)
        _log.debug(f"Generating mappings for parameter {parameter.name} of type {ptype}.")
        if ptype not in valid_types:
            raise RuntimeError(f"ConfigSpace parameter type {ptype} is not accepted by the optimizer.")

        emukit_param, map_to_emu, map_to_cs = param_map[ptype](parameter)
        space.append(emukit_param)
        to_emu.append((parameter.name, map_to_emu))
        to_cs.append((parameter.name, map_to_cs))

    def config_to_cs(x: np.ndarray):
        """ Combines individual parameter parsers into one function for parsing an entire configuration in one go. """
        idx = 0
        conf = {}
        for i in range(len(to_cs)):
            pname, fn = to_cs[i]
            param: Parameter = space[i]
            if param.dimension == 1:
                conf[pname] = fn(x[idx])
            else:
                conf[pname] = fn(x[idx:idx + param.dimension].squeeze())
            idx += param.dimension

        return conf

    # TODO: Create a config_to_emu function once its use case is clear.

    _log.debug(f"Successfully parsed all parameters from ConfigurationSpace {cspace.name}")
    return ParameterSpace(space), to_emu, config_to_cs


def get_init_trajectory_hook(output_dir: Path):
    """
    Generates a before-loop-start hook for recording Information-Theoretic acquisition function MUMBO's real trajectory.
    :param output_dir: Path
        The directory where the trajectory will be stored.
    :return:
    """

    def hook(loop: OuterLoop, loop_state: LoopState):
        """ A function that is called only once, before the optimization begins, to set the stage for recording MUMBO's
        trajectory. """

        path = output_dir / MUMBO_TRAJECTORY_FILE
        with open(path, 'w') as _:
            # Delete old contents in case the file used to exist.
            pass

    return hook


def get_trajectory_hook(output_dir: Path):
    """
    Generates an end-of-iteration hook for recording Information-Theoretic acquisition function MUMBO's real trajectory.
    :param output_dir: Path
        The directory where the trajectory will be stored.
    :return:
    """

    def hook(loop: OuterLoop, loop_state: LoopState):
        """ A function that is called at the end of each BO iteration in order to record the MUMBO trajectory. """

        _log.debug("Executing trajectory hook for MUMBO in iteration %d" % loop_state.iteration)
        # Remember that the MUMBO acquisition was divided by the Cost acquisition before being fed into the optimizer
        acq: MUMBO = loop.candidate_point_calculator.acquisition.numerator
        sampler = RandomDesign(acq.space)
        grid = sampler.get_samples(acq.grid_size)
        # also add the locations already queried in the previous BO steps
        grid = np.vstack([acq.model.X, grid])
        # remove current fidelity index from sample
        grid = np.delete(grid, acq.source_idx, axis=1)
        # Add objective function fidelity index to sample
        idx = np.ones((grid.shape[0])) * acq.target_information_source_index
        grid = np.insert(grid, acq.source_idx, idx, axis=1)
        # Get GP posterior at these points
        fmean, fvar = acq.model.predict(grid)
        mindx = np.argmin(fmean)
        predicted_incumbent = np.delete(grid[mindx], acq.source_idx)
        timestamp = time.time()
        iteration = loop_state.iteration
        with open(output_dir / MUMBO_TRAJECTORY_FILE, "a") as fp:
            fp.writelines([json.dumps(
                {
                    "start_time": None,
                    "finish_time": timestamp,
                    "function_value": None,
                    "fidelity": None,
                    "cost": None,
                    "configuration": predicted_incumbent.tolist(),
                    "info": {
                        "fidelity": None,
                        "function_call": None,
                        "total_time_used": None,
                        "total_objective_costs": None,
                        "iteration": iteration,
                        "predicted_mean": fmean[mindx, 0],
                        "predicted_variance": fvar[mindx, 0]
                    }
                }), "\n"])
        _log.debug("Finished executing trajectory hook.")

    return hook
