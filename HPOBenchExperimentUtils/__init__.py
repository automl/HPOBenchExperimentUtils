import logging

_default_log_format = '[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'
logging.basicConfig(format=_default_log_format, level=logging.WARNING)

_log = logging.getLogger(__name__)

from hpobench.util.example_utils import set_env_variables_to_use_only_one_core  # noqa
set_env_variables_to_use_only_one_core()  # noqa

from HPOBenchExperimentUtils.run_benchmark import run_benchmark  # noqa
from HPOBenchExperimentUtils.validate_benchmark import validate_benchmark  # noqa

# Trying here to import autogluon fixes some weird "Cell is empty" aka Pickle error.
try:
    import autogluon.core as ag
    _log.debug('IMPORTED AUTOGLUON')
except:
    pass
