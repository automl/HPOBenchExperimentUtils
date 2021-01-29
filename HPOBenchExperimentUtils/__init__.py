import logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)
_default_log_format = '[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'

from hpobench.util.example_utils import set_env_variables_to_use_only_one_core  # noqa
set_env_variables_to_use_only_one_core()  # noqa

from HPOBenchExperimentUtils.run_benchmark import run_benchmark  # noqa
from HPOBenchExperimentUtils.validate_benchmark import validate_benchmark  # noqa
