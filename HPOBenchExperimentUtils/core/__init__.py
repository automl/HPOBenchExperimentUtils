WORKER_WAIT_FOR_SCHEDULER_TO_START_IN_S = 600
WORKER_WAIT_FOR_NAMESERVER_TO_START_IN_S = 300
SCHEDULER_PING_WORKERS_INTERVAL_IN_S = 10
SCHEDULER_TIMEOUT_WORKER_DISCOVERY_IN_S = 600

# See Explanation in HPOBenchExperimentUtils/__init__.py
try:
    from HPOBenchExperimentUtils.optimizer.autogluon_optimizer import _obj_fct
except ModuleNotFoundError:
    pass
