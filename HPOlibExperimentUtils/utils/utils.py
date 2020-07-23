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
