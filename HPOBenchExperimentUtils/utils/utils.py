import logging
import signal
from contextlib import contextmanager
from typing import Union
import numpy as np

_log = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """ Simple time limit enforcer script. We use it to make sure that each configuration only runs a given time. """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_mandatory_optimizer_setting(settings_dict: dict, setting_name: str, err_msg: str = None):
    """ Convenience function that tries to fetch a given string from the settings dictionary and raises an error if it
    is not found. """

    if err_msg is None:
        err_msg = "The optimizer settings must include '%s'." % setting_name

    try:
        return settings_dict[setting_name]
    except KeyError as e:
        raise KeyError(err_msg) from e


def standard_rng_init(rng: Union[np.random.RandomState, int, None]):
    if isinstance(rng, np.random.RandomState):
        return rng
    else:
        return np.random.RandomState(rng)
