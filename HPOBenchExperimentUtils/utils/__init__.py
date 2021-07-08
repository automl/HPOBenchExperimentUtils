import logging

_log = logging.getLogger(__name__)

MAXINT = 2 ** 31 - 1
PING_OPTIMIZER_IN_S = 1

# Define constants
RUNHISTORY_FILENAME = 'hpobench_runhistory.txt'
TRAJECTORY_V1_FILENAME = 'hpobench_trajectory_v1.txt'
TRAJECTORY_V2_FILENAME = 'hpobench_trajectory_v2.txt'
TRAJECTORY_V3_FILENAME = 'hpobench_trajectory_v3.txt'


VALIDATED_RUNHISTORY_FILENAME = 'hpobench_runhistory_validation.txt'
VALIDATED_TRAJECTORY_V1_FILENAME = 'hpobench_trajectory_validated_v1.txt'
VALIDATED_TRAJECTORY_V2_FILENAME = 'hpobench_trajectory_validated_v2.txt'
VALIDATED_TRAJECTORY_V3_FILENAME = 'hpobench_trajectory_validated_v3.txt'

RESOURCE_FILENAME = 'hpobench_used_resources.json'
