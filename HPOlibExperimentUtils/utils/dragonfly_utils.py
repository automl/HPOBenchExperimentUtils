from math import exp, log, floor
from typing import List, Dict, Tuple, Union, Callable
from pathlib import Path
from argparse import Namespace
import logging
import os, uuid
import numpy as np

logger = logging.getLogger('Dragonfly Utils')

# -------------------------------Begin code adapted directly from the dragonfly repo------------------------------------

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter, Constant

from dragonfly.exd.exd_utils import get_unique_list_of_option_args
from dragonfly.utils.option_handler import get_option_specs
# Get options
from dragonfly.opt.ga_optimiser import ga_opt_args
from dragonfly.opt.gp_bandit import get_all_euc_gp_bandit_args, \
    get_all_cp_gp_bandit_args, get_all_mf_euc_gp_bandit_args, \
    get_all_mf_cp_gp_bandit_args
from dragonfly.opt.random_optimiser import euclidean_random_optimiser_args, \
    mf_euclidean_random_optimiser_args, \
    cp_random_optimiser_args, mf_cp_random_optimiser_args
from dragonfly.opt.multiobjective_gp_bandit import get_all_euc_moo_gp_bandit_args, \
    get_all_cp_moo_gp_bandit_args
from dragonfly.opt.random_multiobjective_optimiser import \
    euclidean_random_multiobjective_optimiser_args, \
    cp_random_multiobjective_optimiser_args
from dragonfly.utils.option_handler import load_options


_dragonfly_args = [
    # get_option_specs('config', False, None, 'Path to the json or pb config file. '),
    # get_option_specs('options', False, None, 'Path to the options file. '),
    get_option_specs('max_or_min', False, 'max', 'Whether to maximise or minimise. '),
    get_option_specs('max_capital', False, -1.0,
                     'Maximum capital (available budget) to be used in the experiment. '),
    get_option_specs('capital_type', False, 'return_value',
                     'Maximum capital (available budget) to be used in the experiment. '),
    get_option_specs('is_multi_objective', False, 0,
                     'If True, will treat it as a multiobjective optimisation problem. '),
    get_option_specs('opt_method', False, 'bo',
                     ('Optimisation method. Default is bo. This should be one of bo, ga, ea, direct, ' +
                      ' pdoo, or rand, but not all methods apply to all problems.')),
    get_option_specs('report_progress', False, 'default',
                     ('How to report progress. Should be one of "default" (prints to stdout), ' +
                      '"silent" (no reporting), or a filename (writes to file).')),
]


def _get_command_line_args():
    """ Returns all arguments for the command line. """
    ret = _dragonfly_args + \
          ga_opt_args + \
          euclidean_random_optimiser_args + cp_random_optimiser_args + \
          mf_euclidean_random_optimiser_args + mf_cp_random_optimiser_args + \
          get_all_euc_gp_bandit_args() + get_all_cp_gp_bandit_args() + \
          get_all_mf_euc_gp_bandit_args() + get_all_mf_cp_gp_bandit_args() + \
          euclidean_random_multiobjective_optimiser_args + \
          cp_random_multiobjective_optimiser_args + \
          get_all_euc_moo_gp_bandit_args() + get_all_cp_moo_gp_bandit_args()
    return get_unique_list_of_option_args(ret)


# ---------------------------------End code adapted directly from the dragonfly repo------------------------------------

from dragonfly.parse.config_parser import load_parameters
from dragonfly.exd.cp_domain_utils import load_config


def load_dragonfly_options(options: Dict, config: Dict) -> Tuple[Namespace, Dict]:
    options = load_options(_get_command_line_args(), partial_options=options, cmd_line=False)
    config = load_config(load_parameters(config))
    return options, config


# TODO: Add more hyperparameter types and type-specific handlers.

def _handler_unknown(hyp):
    raise RuntimeError("No valid handler available for hyperparameter of type %s" % type(hyp))


def _handle_uniform_float(hyper: UniformFloatHyperparameter) -> Tuple[Dict, Callable, Callable]:
    """
    Handles the mapping of ConfigSpace.UniformFloatHyperparameter objects to dragonfly's 'float' parameters.
    Caveats:
        - Dragonfly does not support sampling on a log scale, therefore this mapping will instead ask dragonfly to
          uniformly sample values in the range [log(lower), log(upper)], and then forward the exponentiated sampled
          values to the objective function.
        - It is assumed that the costs are directly proportional to the sampled value, such that the minimum value
          corresponds to a cost of 0 and the maximum value corresponds to a cost of 1.
    """
    domain = {
        'name': hyper.name,
        'type': 'float',
        'min': log(hyper.lower) if hyper.log else hyper.lower,
        'max': log(hyper.upper) if hyper.log else hyper.upper
    }

    parser = (lambda x: exp(x)) if hyper.log else (lambda x: x)
    # Here, x is in the mapped space!
    cost = lambda x: (x - domain['min']) / (domain['max'] - domain['min'])
    return domain, parser, cost


def _handle_uniform_int(hyper: UniformFloatHyperparameter) -> Tuple[Dict, Callable, Callable]:
    """
    Handles the mapping of ConfigSpace.UniformFloatHyperparameter objects to dragonfly's 'int' parameters.
    Caveats:
        - Dragonfly does not support sampling on a log scale, therefore this mapping will instead ask dragonfly to
          uniformly sample integers in the range [floor(log(lower)), floor(log(upper))], and then forward the
          exponentiated sampled values to the objective function.
        - It is assumed that the costs are a directly proportional to the sampled value, such that the minimum value
          corresponds to a cost of 0 and the maximum value corresponds to a cost of 1.
    """
    domain = {
        'name': hyper.name,
        'type': 'int',
        'min': floor(log(hyper.lower)) if hyper.log else hyper.lower,
        'max': floor(log(hyper.upper)) if hyper.log else hyper.upper
    }

    parser = (lambda x: exp(x)) if hyper.log else (lambda x: x)
    # Here, x is in the mapped space!
    cost = lambda x: (x - domain['min']) / (domain['max'] - domain['min'])
    return domain, parser, cost


def _handle_categorical(hyper: CategoricalHyperparameter) -> Tuple[Dict, Callable, Callable]:
    """
    Handles the mapping of ConfigSpace.CategoricalHyperparameter objects to dragonfly's 'discrete' parameters.
    Caveats:
        - Dragonfly cannot handle non-uniform item weights.
        - Dragonfly internally stores and samples all items as strings. An attempt will be made to automatically handle
          the type conversion by inferring and storing the data type using a call to type(), hence it is expected that
          the data types of the items support direct conversion to and from strings.
        - It is assumed that each individual choice incurs exactly the same cost, 1/N, where N is the number of choices.
    """

    if not isinstance(hyper.choices, (list, tuple)):
        raise TypeError("Expected choices to be either list or tuple, received %s" % str(type(hyper.choices)))

    n = len(hyper.choices)
    choices = tuple(hyper.choices)
    probs = hyper.probabilities
    if probs is None:
        cumprobs = np.repeat(1. / n, n)
    else:
        cumprobs = np.cumsum(probs)
        assert cumprobs.shape[0] == n, "The number of cumulative probability values should match the number of choices."
        assert cumprobs[-1] == 1., "The given probability values have not been normalized."

    def _choose(pval: float):
        return np.asarray(pval <= cumprobs).nonzero()[0][0]

    domain = {
        'name': hyper.name,
        'type': 'float',
        'min': 0.0,
        'max': 1.0,
    }

    parser = lambda x: choices[_choose(x)]
    cost = lambda x: probs[_choose(x)]
    return domain, parser, cost


def _handle_ordinal(hyper: OrdinalHyperparameter) -> Tuple[Dict, Callable, Callable]:
    """
    Handles the mapping of ConfigSpace.OrdinalHyperparameter objects to dragonfly's 'discrete_numeric' parameters.
    Caveats:
        - The only difference between an Ordinal and a Categorical is the meta-information of item ordering, which is
          not useful for dragonfly in any case, therefore dragonfly is only provided indices to an internally stored
          ordered sequence.
        - It is assumed that the costs are directly proportional to the index location of the sampled value, such that
          the item with index 0 or the first item in the sequence incurs a cost of 0 and the last item incurs a
          cost of 1.
    """

    sequence = hyper.sequence
    if not isinstance(sequence, (list, tuple)):
        raise TypeError("Expected sequence to be either list or tuple, received %s" % str(type(sequence)))

    n = len(sequence) - 1
    domain = {
        'name': hyper.name,
        'type': 'discrete',
        'items': "-".join([str(i) for i in range(n+1)])
    }

    parser = lambda x: sequence[int(x)]
    cost = lambda x: x / n
    return domain, parser, cost


_handlers = {
    UniformFloatHyperparameter: _handle_uniform_float,
    UniformIntegerHyperparameter: _handle_uniform_int,
    CategoricalHyperparameter: _handle_categorical,
    OrdinalHyperparameter: _handle_ordinal
}


def _configspace_to_dragonfly(params: List[Hyperparameter]) -> Tuple[Dict, List, List]:
    dragonfly_dict = {}
    parsers = []
    costs = []
    for param in params:
        d, p, c = _handlers.get(type(param), _handler_unknown)(param)
        dragonfly_dict[param.name] = d
        parsers.append((param.name, p))
        costs.append(c)

    return dragonfly_dict, parsers, costs


def configspace_to_dragonfly(domain_cs: ConfigurationSpace, name="hpolib_benchmark",
                             fidely_cs: ConfigurationSpace = None) -> \
        Tuple[Dict, List, Union[List, None], Union[List, None]]:

    domain, domain_parsers, _ = _configspace_to_dragonfly(domain_cs.get_hyperparameters())
    out = {'name': name, 'domain': domain}
    if fidely_cs:
        # fidelity_space, fidelity_parsers = _generate_xgboost_fidelity_space(fidely_cs)
        fidelity_space, fidelity_parsers, fidelity_costs = _configspace_to_dragonfly(fidely_cs.get_hyperparameters())
        out['fidel_space'] = fidelity_space
        # out['fidel_to_opt'] = [fidel['max'] for _, fidel in fidelity_space.items()]
        out['fidel_to_opt'] = [param.default_value for param in fidely_cs.get_hyperparameters()]
        logger.debug("Generated fidelity space %s\nfidelity optimization taret: %s" %
                     (fidelity_space, out['fidel_to_opt']))
        return out, domain_parsers, fidelity_parsers, fidelity_costs
    else:
        return out, domain_parsers, None, None
    # TODO: Add support for converting constraints


def generate_trajectory(history: Namespace, save_file: Path, is_cp=False, history_file=None):
    """
    Given the history generated by a call to minimise_function in dragonfly, generates a SMAC-like trajectory and
    saves it as the given file. The parameter save_file should be the full path of the filename to which the history is
    to be saved. The is_cp flag indicates that a Cartesian Product space was used, thus affecting the output format. If
    a history_file is specified, the dragonfly run history will be dumped to that file.
    """
    if history_file is not None:
        history_file = Path(history_file)
        if not history_file.is_absolute():
            history_file.expanduser().resolve()
        recorded_history = []
        save_history = True
    else:
        save_history = False

    trajectories = []
    incumbent = {
        "cpu_time": float(0),
        "wallclock_time": float(0),
        "evaluations": int(0),
        "cost": float('inf'),
        "incumbent": None,
        "origin": "xxx"
    }
    update = False

    for qinfo in history.query_qinfos:
        # Remember, dragonfly maximizes.
        # In the history namespace, query_true_vals refers to the values used for maximization, and query_vals refers
        # to the actual value returned from the objective function. This means that if the optimizer was told to
        # minimize instead of maximize, query_true_vals will be the negated query_vals. However, the corresponding
        # fields in each query_qinfo do not follow this convention and always contain the value used for maximization.

        if -qinfo.val < incumbent["cost"]:
            incumbent = {
                "cpu_time": qinfo.receive_time,
                "wallclock_time": qinfo.receive_time,
                "evaluations": qinfo.step_idx,
                "cost": -qinfo.val,
                "incumbent": [list(pt) for pt in qinfo.point] if is_cp else list(qinfo.point),
                "origin": "xxx" if not hasattr(qinfo, "curr_acq") else qinfo.curr_acq
            }
            update = True

        if not trajectories or update:
            trajectories.append(incumbent)
            update = False

        if save_history:
            recorded_history.append({
                "cpu_time": qinfo.receive_time,
                "wallclock_time": qinfo.receive_time,
                "evaluations": qinfo.step_idx,
                "cost": -qinfo.val,
                "incumbent": [list(pt) for pt in qinfo.point] if is_cp else list(qinfo.point),
                "origin": "xxx" if not hasattr(qinfo, "curr_acq") else qinfo.curr_acq
            })
    import json
    with open(save_file, "w") as f:
        f.write("\n".join([json.dumps(t, indent=4) for t in trajectories]))
        # json.dump(trajectories, f, indent=4)

    if save_history:
        with open(history_file, 'w') as fp:
            json.dump(recorded_history, fp, indent=4)

    print("Finished writing trajectories file.")


def change_cwd(tries=5):
    if tries <= 0:
        raise RuntimeError("Could not create random temporary dragonfly directory due to timeout.")

    try:
        tmp_dir = Path("/tmp") / "dragonfly" / str(uuid.uuid4())
        tmp_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        change_cwd(tries=tries - 1)
    else:
        os.chdir(tmp_dir)
        logger.debug("Switched to temporary directory %s" % str(tmp_dir))
    return
