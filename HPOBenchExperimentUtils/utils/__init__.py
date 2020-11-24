import logging

_log = logging.getLogger(__name__)

MAXINT = 2 ** 31 - 1
PING_OPTIMIZER_IN_S = 0.25

# Define constants
RUNHISTORY_FILENAME = 'hpobench_runhistory.txt'
TRAJECTORY_FILENAME = 'hpobench_trajectory.txt'
VALIDATED_RUNHISTORY_FILENAME = 'hpobench_runhistory_validation.txt'