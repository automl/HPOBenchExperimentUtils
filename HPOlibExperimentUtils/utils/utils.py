import logging
import signal
from contextlib import contextmanager

logger = logging.getLogger('Utils')


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


def get_main_fidelity(fidelity_space, settings):
    """Helper function to get the main fidelity from a fidelity space. """
    if len(fidelity_space.get_hyperparameters()) > 1 and 'main_fidelity' not in settings:
        raise ValueError('Ok something went wrong. Please specify a main fidelity in the benchmark settings')

    if 'main_fidelity' in settings:
        main_fidelity = settings['main_fidelity']
        fidelity = fidelity_space.get_hyperparameter(main_fidelity)
    else:
        fidelity = fidelity_space.get_hyperparameters()[0]
    return fidelity