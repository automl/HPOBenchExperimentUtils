import logging
import json
import sys
import threading

import Pyro4
import Pyro4.util
import Pyro4.naming
import Pyro4.errors

from typing import Union, Dict
from time import time, sleep
from pathlib import Path

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from HPOBenchExperimentUtils.core import WORKER_WAIT_FOR_SCHEDULER_TO_START_IN_S, \
    WORKER_WAIT_FOR_NAMESERVER_TO_START_IN_S
from HPOBenchExperimentUtils.core.daemon_object import DaemonObject
from HPOBenchExperimentUtils.core.record import Record
from HPOBenchExperimentUtils.utils.runner_utils import load_benchmark, get_benchmark_names, \
    transform_unknown_params_to_dict, get_benchmark_settings

sys.excepthook = Pyro4.util.excepthook


class Worker(DaemonObject):

    def __init__(self, worker_id: int, run_id: Union[int, str], ns_ip: str, ns_port: int, debug: bool = False):
        # Unique name of the worker
        self.worker_id = f'Worker_{run_id}_{worker_id}_{threading.get_ident()}'

        self.logger = logging.getLogger(f'{self.worker_id}')
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # All workers and the scheduler should have the same run_id
        # TODO: Actually, we dont need a run id, since we have a unique nameserver for each run
        self.run_id = run_id
        self.is_working = False

        self.benchmark = None  # type: Union[AbstractBenchmarkClient, None]
        self.benchmark_settings = None  # type: [Dict, None]

        super(Worker, self).__init__(ns_ip=ns_ip, ns_port=ns_port, registration_name=self.worker_id, logger=self.logger)

        self.logger.debug(f'The worker is successfully started. It has the ID: {worker_id}. The nameserver is reachable'
                          f'here: {ns_ip}:{ns_port}')

    def start_up(self, benchmark_settings: Dict, benchmark_params: Dict, rng, use_local=False):
        """
        Load the benchmark object. This step may take some time. Make sure that the scheduler waits long enough for the
        worker to start.

        Parameters
        ----------
        benchmark_settings : Dict
            Settings like the container name, time limit, etc.
        benchmark_params : Dict
            If additional parameters are given, e.g. the task-id for the XGBoostBenchmark, we can pass them to the
            benchmark.
        rng : np.random.RandomState, int
        use_local : bool
            If True: use the local benchmark version and not the containerized.

        """
        self.benchmark_settings = benchmark_settings

        # Load and instantiate the benchmark
        benchmark_obj = load_benchmark(benchmark_name=benchmark_settings['import_benchmark'],
                                       import_from=benchmark_settings['import_from'],
                                       use_local=use_local)

        if not use_local:
            from hpobench import config_file
            benchmark_params['container_source'] = config_file.container_source
        self.benchmark = benchmark_obj(rng=rng, **benchmark_params)

    def evaluate_configuration(self, configuration: Dict, fidelity: Dict, **kwargs) -> Dict:
        """
        Call the objective_function_test of the benchmark and evaluate the configuration with a given fidelity.
        The record object contains all necessary information. It is also used in the bookkeeper, which can collect more
        information. However, we don't need all fields, so we leave some empty.

        Parameters
        ----------
        configuration: Dict
        fidelity: Dict
        kwargs : Any

        Returns
        -------
        Dict
        """
        self.logger.debug(f'Start evaluating the configuration {configuration} with fidelity {fidelity}.'
                          f'Additional Arguments are: {kwargs}')

        record = Record()
        record.start_time = time()

        result_dict = self.benchmark.objective_function_test(configuration, fidelity=fidelity, **kwargs)

        record.finish_time = time()
        record.function_value = result_dict['function_value']
        record.cost = result_dict['cost']
        record.info = result_dict['info']
        if kwargs is not None or (isinstance(kwargs, Dict) and len(kwargs) != 0):
            record.info['AdditionalArguments'] = kwargs
        record.configuration = configuration
        try:
            record.fidelity = result_dict['info']['fidelity']
        except KeyError:
            record.fidelity = fidelity

        return record.get_dictionary()

    @Pyro4.expose
    def is_alive(self):
        """
        Check if the worker is still alive and running. Also, check the background thread. If the background thread is
        crashed, then an AttributeError is thrown.

        Returns
        -------
        bool
        """

        try:
            thread_state = self.thread.is_alive()
        except AttributeError:
            thread_state = False

        self.logger.debug(
            f'The worker state: thread_is_running: {self.thread_is_running} - thread responsible: {thread_state}')
        return self.thread_is_running and thread_state

    def run(self, wait_for_scheduler_to_start_in_s: int = WORKER_WAIT_FOR_SCHEDULER_TO_START_IN_S) -> None:
        """
        Main Function of the worker.  First, get the address of the scheduler. Then, aks the scheduler, as long as it
        has some configurations to validate left, for a task.
        Process the configuration and send the result back to the scheduler.

        Parameters
        ----------
        wait_for_scheduler_to_start_in_s : Optional[int]
            If the worker can not find the scheduler, the scheduler may still start up. The worker should wait for
            a given time. The function throws a NamingError, when the time limit is reached.

            The time limit defaults to the globally defined time limit.
        """
        self.logger.debug('Start the run.')
        with Pyro4.locateNS(host=self.ns_ip, port=self.ns_port) as nameserver:
            self.logger.debug('Nameserver located.')

            scheduler_uri = self._get_scheduler_uri(nameserver, wait_for_scheduler_to_start_in_s)

        self.working_loop(scheduler_uri)

        self.logger.debug('Quit Main Loop.')

    def working_loop(self, scheduler_uri: Pyro4.core.URI):
        """
        Ask the scheduler for a configuration, validate it, and register the results.

        Parameters
        ----------
        scheduler_uri : URI
            Address of the scheduler.

        """
        start_time = time()
        wallclock_time_limit = self.benchmark_settings.get('time_limit_in_s')

        while time() - start_time <= wallclock_time_limit:
            self.is_working = True

            # First, request a configuration (fidelity, additional)
            with Pyro4.Proxy(scheduler_uri) as scheduler:
                # TODO: Handle Timeout? Use async (Future)?
                configuration, fidelity, additional, msg_contains_item = scheduler.get_content(self.uri)

            self.logger.debug(f'Received: {configuration}, {fidelity}, {additional}.'
                              f' Item received: {msg_contains_item}')

            # First, check if we received a configuration. If the value is false, we know that the scheduler's queue is
            # empty and all configurations have been validated, so we can stop the worker.
            if not msg_contains_item:
                self.logger.info('The scheduler has no configurations left. Worker is going to shut down.')
                break

            # Validate the configuration
            record_dict = self.evaluate_configuration(configuration, fidelity, **additional)
            self.logger.debug(f'Evaluated configuration: {record_dict}')

            # Send the result back to the scheduler
            with Pyro4.Proxy(scheduler_uri) as scheduler:
                self.logger.debug('Trying to send the result back to the scheduler.')
                successful = scheduler.register_result(configuration, fidelity, additional, record_dict)
                self.logger.debug(f'The results was successfully sent to the scheduler? {successful}')

            self.is_working = False
            self.logger.debug(f'The worker has finished its computations. It is now trying to request a new '
                             f'configuration')
        else:
            self.logger.warning(f'Timelimit of {wallclock_time_limit} reached.')

    def _get_scheduler_uri(self, nameserver, timeout: int):
        """ Establishes a connection to the nameserver. """

        scheduler_discorvery_start_time = time()
        schedulers = []

        while len(schedulers) == 0 and (time() - scheduler_discorvery_start_time) <= timeout:

            self.logger.debug(f'Trying to find the scheduler with the name Scheduler_{self.run_id}.')
            schedulers = nameserver.list(prefix=f'Scheduler_{self.run_id}')

            if len(schedulers) != 0:
                self.logger.debug('Found the scheduler.')
                break

            self.logger.debug('The scheduler is still not online. Sleep for 5 seconds and try again.')
            sleep(5)
        else:
            self.logger.exception('No scheduler was found. Please make sure that the scheduler can be found on '
                                  f'{self.ns_ip}:{self.ns_port}. And it has the same run id ({self.run_id}')
            raise Pyro4.errors.NamingError()

        assert len(schedulers) == 1, 'Multiple schedulers are found (!?)'

        scheduler_uri = schedulers.get(f'Scheduler_{self.run_id}')
        self.logger.info(f'Scheduler located. Uri is {scheduler_uri}.')

        return scheduler_uri

    def __del__(self):
        self.logger.info('Calling the Shutdown method')
        self.benchmark.__del__()
        super(Worker, self).__del__()


def start_worker(benchmark: str,
                 credentials_dir: Union[Path, str],
                 run_id: int,
                 worker_id: int,
                 rng: int,
                 use_local: Union[bool, None] = False,
                 debug: Union[bool, None] = False,
                 **benchmark_params: Dict):

    if benchmark_params is None:
        benchmark_params = {}
    logger = logging.getLogger(f'{worker_id}')
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info('Call Worker Main')
    logger.debug(f'Benchmark Params: {vars(benchmark_params)}')

    benchmark_settings = get_benchmark_settings(benchmark)

    # Load the nameserver address
    credentials_dir = Path(credentials_dir)
    credentials_file = credentials_dir / f'HPBenchExpUtils_pyro4_nameserver_{run_id}.json'

    # Wait X seconds for the scheduler to start the nameserver
    cred_file_discovery_start = time()
    logger.debug('Start discovering the nameserver credentials.')
    while time() - cred_file_discovery_start <= WORKER_WAIT_FOR_NAMESERVER_TO_START_IN_S:
        if credentials_file.exists():
            break
        logger.debug('Could not find the nameserver credentials file. '
                     'Sleep for 5 seconds and try again')
        sleep(5)
    else:
        raise FileNotFoundError('Could not find the nameserver credentials.')

    logger.debug('Found the nameserver credentials.')

    with credentials_file.open('r') as fh:
        ns_ip, ns_port = json.load(fh)

    logger.info('Credentials loaded from file. Going to start the worker.')

    with Worker(run_id=run_id, worker_id=worker_id, ns_ip=ns_ip, ns_port=ns_port, debug=debug) as worker:
        worker.start_up(benchmark_settings, benchmark_params, rng, use_local)
        worker.run()

    logger.info('Worker has finished its run.')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--credentials_dir', type=str, required=True,
                        help='The address of the nameserver is stored here')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Unique name of the run')
    parser.add_argument('--worker_id', type=int, required=True,
                        help='Unique name of the worker')
    parser.add_argument('--rng', required=False, default=0, type=int)
    parser.add_argument('--use_local', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False, help="When given, enables debug mode logging.")

    args, unknown = parser.parse_known_args()

    return args, unknown


if __name__ == '__main__':
    args, unknown = parse_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    start_worker(args.benchmark,
                 args.credentials_dir,
                 args.run_id,
                 args.worker_id,
                 args.rng,
                 args.use_local,
                 args.debug,
                 **benchmark_params)
