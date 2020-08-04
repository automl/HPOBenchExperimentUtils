from math import log10, log2
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from argparse import Namespace
import logging

logger = logging.getLogger('Dragonfly Utils')

# -------------------------------Begin code adapted directly from the dragonfly repo------------------------------------

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter, UniformFloatHyperparameter

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


def _handle_uniform_float(hyper: UniformFloatHyperparameter):
    domain = {
        'name': hyper.name,
        'type': 'float',
        'min': log10(hyper.lower) if hyper.log else hyper.lower,
        'max': log10(hyper.upper) if hyper.log else hyper.upper
    }

    parser = (lambda x: 10 ** x) if hyper.log else (lambda x: x)
    return domain, parser


_handlers = {
    UniformFloatHyperparameter: _handle_uniform_float
}


def _configspace_to_dragonfly_domain(hypers: List[Hyperparameter]) -> Tuple[Dict, List]:
    domain = {}
    parser = []
    for hyp in hypers:
        d, p = _handlers.get(type(hyp), _handler_unknown)(hyp)
        domain[hyp.name] = d
        parser.append((hyp.name, p))

    return domain, parser


# TODO: Switch to ConfigurationSpace objects
def _generate_xgboost_fidelity_space(fidel_dict: Dict) -> Tuple[Dict, List]:
    """
    Given a dict of fidelities read from experiment_settings.json for the xgboost benchmark, returns an appropriate
    ConfigurationSpace object. Temporary hack until a more stable solution is found.
    """
    fspace = {}
    parsers = []

    key = 'subsample'
    if key in fidel_dict:
        fidel = {
            'name': key,
            'type': 'float',
            'min': 0.1,
            'max': 1.0
        }
        parser = lambda x: x
        cost = lambda x: 2
        fspace[key] = fidel
        parsers.append((key, parser, cost))

    key = 'n_estimators'
    if key in fidel_dict:
        log = False if fidel_dict[key].lower() == 'linear' else True
        fidel = {
            'name': key,
            'type': 'int',
            'min': int(log2(1)) if log else 1,
            'max': int(log2(128)) if log else 128
        }
        parser = (lambda x: 2 ** x) if log else (lambda x: x)
        cost = lambda x: ((x - fidel['min']) / (fidel['max'] - fidel['min']))
        fspace[key] = fidel
        parsers.append((key, parser, cost))

    return fspace, parsers


# TODO: Switch fidelities to ConfigurationSpace objects
def configspace_to_dragonfly(domain_cs: ConfigurationSpace, name="hpolib_benchmark",
                             fidely_cs: Dict = None) -> Tuple[Dict, List, Union[List, None]]:
    domain, domain_parsers = _configspace_to_dragonfly_domain(domain_cs.get_hyperparameters())
    out = {'name': name, 'domain': domain}
    if fidely_cs:
        fidelity_space, fidelity_parsers = _generate_xgboost_fidelity_space(fidely_cs)
        out['fidel_space'] = fidelity_space
        out['fidel_to_opt'] = [fidel['max'] for _, fidel in fidelity_space.items()]
        logger.debug("Generated fidelity space %s\nfidelity optimization taret: %s" % (fidelity_space, out['fidel_to_opt']))
        return out, domain_parsers, fidelity_parsers
    else:
        return out, domain_parsers, None
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
        f.write("\n".join([json.dumps(t) for t in trajectories]))
        # json.dump(trajectories, f, indent=4)

    if save_history:
        with open(history_file, 'w') as fp:
            json.dump(recorded_history, fp, indent=4)

    print("Finished writing trajectories file.")
