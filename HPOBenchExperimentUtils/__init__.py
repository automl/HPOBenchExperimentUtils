import logging

_default_log_format = '[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'
logging.basicConfig(format=_default_log_format, level=logging.WARNING)

_log = logging.getLogger(__name__)

from hpobench.util.example_utils import set_env_variables_to_use_only_one_core  # noqa
set_env_variables_to_use_only_one_core()  # noqa

from HPOBenchExperimentUtils.run_benchmark import run_benchmark  # noqa
from HPOBenchExperimentUtils.validate_benchmark import validate_benchmark  # noqa

# We have encountered a weird error when using autogluon. The optimizer has crashed with either a "Cell is empty" or a
# "PicklingError: Can't pickle <class XY>: it's not the same object as XY"-error.
# We were able to trace it back to some interaction between dill and cloudpickle (used in the distributed package).
# Importing autogluon here seems to fix the error.
try:
    # import autogluon.core as ag
    from HPOBenchExperimentUtils.optimizer.autogluon_optimizer import _obj_fct
    _log.debug('Autogluon succesfully imported')
except ModuleNotFoundError:
    pass

try:
    import ray
    from ray import tune as tune
    _log.debug('Ray succesfully imported')
except ModuleNotFoundError:
    pass
