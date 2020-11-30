import logging
from typing import Tuple, Callable, Sequence
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
import ConfigSpace as cs

_log = logging.getLogger(__name__)


def _handle_uniform_float(param: cs.UniformFloatHyperparameter) -> Tuple[ContinuousParameter, Callable, Callable]:
    """ Generate a mapping for a UniformFloatHyperparameter object. """
    min_val, max_val = param.lower, param.upper
    if param.log:
        min_val = log(min_val)
        max_val = log(max_val)
        map_to_emu = lambda x: log(x)
        map_to_cs = lambda x: exp(x)
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
