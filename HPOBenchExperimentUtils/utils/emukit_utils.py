import logging
from typing import Tuple, Callable, Sequence
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
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
        map_to_emu = lambda x: log(x)
        map_to_cs = lambda x: min(param.upper, max(param.upper, exp(x)))
    else:
        map_to_emu = lambda x: x
        map_to_cs = lambda x: x
    emukit_param = ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)

    return emukit_param, map_to_emu, map_to_cs


def _handle_uniform_int(param: cs.UniformIntegerHyperparameter) -> \
        Tuple[ContinuousParameter, Callable, Callable]:
    """ Generate a mapping for a UniformIntegerHyperparameter object. """

    if param.log:
        min_val, max_val = log(param.lower), log(param.upper)
        map_to_emu = lambda x: log(x)
        map_to_cs = lambda x: round(exp(x))
    else:
        min_val, max_val = param.lower, param.upper
        map_to_emu = lambda x: x
        map_to_cs = lambda x: round(x)

    emukit_param = ContinuousParameter(name=param.name, min_value=min_val, max_value=max_val)
    return emukit_param, map_to_emu, map_to_cs


param_map = {
    cs.UniformFloatHyperparameter: _handle_uniform_float,
    cs.UniformIntegerHyperparameter: _handle_uniform_int
}


def generate_space_mappings(cspace: cs.ConfigurationSpace) -> \
        Tuple[ParameterSpace, Sequence[Tuple[str, Callable]], Sequence[Tuple[str, Callable]]]:
    """ Map a ConfigSpace.ConfigurationSpace object to an emukit compatible version and generate the relevant mappings
    to work across the two spaces. """

    space = []
    to_emu = []
    to_cs = []
    for parameter in cspace.get_hyperparameters():
        emukit_param, map_to_emu, map_to_cs = param_map[type(parameter)](parameter)
        space.append(emukit_param)
        to_emu.append((parameter.name, map_to_emu))
        to_cs.append((parameter.name, map_to_cs))

    return ParameterSpace(space), to_emu, to_cs


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
            fp.write(json.dumps(
                {
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "configuration": predicted_incumbent.tolist(),
                    "gp_prediction": np.asarray([fmean[mindx], fvar[mindx]]).squeeze().tolist()
                },
                indent=4
            ))
        _log.debug("Finished executing trajectory hook.")

    return hook

class SmarterInformationSourceParameter(InformationSourceParameter):
    """ Because the base implementation is not very compatible with FABOLAS. """

    def __init__(self, n_sources: int, start_ind: int = 0) -> None:
        """
        :param n_sources: Number of information sources in the problem
        """
        stop_ind = start_ind + n_sources
        super(InformationSourceParameter, self).__init__('source', np.arange(start_ind, stop_ind))
