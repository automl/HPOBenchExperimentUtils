import logging

import Pyro4
import Pyro4.util
import sys
import Pyro4.naming
import Pyro4.errors
import threading
import json
from pathlib import Path
from typing import Union, Dict, Any
from time import time, sleep

sys.excepthook = Pyro4.util.excepthook

from HPOBenchExperimentUtils.core.DaemonObject import DaemonObject

WAIT_FOR_SCHEDULER_TO_START_IN_S = 60


class Worker(DaemonObject):
    def __init__(self, worker_id: int, run_id: Union[int, str], ns_ip: str, ns_port: int):
        # Unique name of the worker
        self.worker_id = f'Worker_{run_id}_{worker_id}_{threading.get_ident()}'

        self.logger = self.__setup_logger(level=logging.DEBUG)

        # All workers and the scheduler should have the same run_id
        # TODO: Actually, we dont need a run id, since we have a unique nameserver for each run
        self.run_id = run_id
        self.is_working = False

        super(Worker, self).__init__(ns_ip=ns_ip, ns_port=ns_port, registration_name=self.worker_id, logger=self.logger)

        self.logger.debug(f'The worker is successfully started. It has the ID: {worker_id}. The nameserver is reachable'
                          f'here: {ns_ip}:{ns_port}')

        self.start_up()

    def start_up(self):
        raise NotImplementedError()

    def evaluate_configuration(self, configuration: Dict, fidelity: Dict) -> Dict:
        """
        Call the objective function of the benchmark and evaluate the configuration with a given fidelity.

        Parameters
        ----------
        configuration: Dict
        fidelity: Dict

        Returns
        -------
        Dict
        """

        self.logger.debug(f'Start evaluating the configuration {configuration} with fidelity {fidelity}')
        # TODO: remove dummy return value
        return dict(function_value=0,
                    info=dict(configuration=configuration,
                              fidelity=fidelity))

    @Pyro4.expose
    def is_alive(self):
        """
        Check if the worker is still alive and running. Also, checks the background thread. If the background thread is
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

    def run(self, wait_for_scheduler_to_start_in_s=None):
        # TODO: Move this in a with statement or Pyro_release!
        self.logger.debug('Start the run.')
        with Pyro4.locateNS(host=self.ns_ip, port=self.ns_port) as nameserver:
            self.logger.debug('Nameserver located.')

            # locate Scheduler
            scheduler_uri = self._get_scheduler_uri(nameserver, wait_for_scheduler_to_start_in_s)

            self.working_loop(scheduler_uri)

    def working_loop(self, scheduler_uri):
        while True:
            self.is_working = True

            # request configuration
            with Pyro4.Proxy(scheduler_uri) as scheduler:
                # TODO: Timeout? Future?
                configuration, fidelity, msg_contains_item = scheduler.get_content(self.uri)
            self.logger.debug(f'Received: {configuration} and {fidelity}. Items received: {msg_contains_item}')

            # First, check if we received a configuration.
            # Then, we could start to gracefully stop the worker
            if not msg_contains_item:
                self.logger.info('The scheduler has no configurations left. '
                                 'Worker is going to shut down.')
                break

            # work on configuration
            result_dict = self.evaluate_configuration(configuration, fidelity)
            self.logger.debug(f'Evaluated configuration: {result_dict}')

            # send result to scheduler back
            with Pyro4.Proxy(scheduler_uri) as scheduler:
                self.logger.debug('Trying to send the result back to the scheduler.')
                successful = scheduler.register_result(configuration, fidelity, result_dict)
                self.logger.info(f'The results was successfully sent to the scheduler? {successful}')

            self.is_working = False

            self.logger.info(f'The worker has finished its computations. It is now trying to request a new '
                             f'configuration')

    def _get_scheduler_uri(self, nameserver, timeout):

        timeout = WAIT_FOR_SCHEDULER_TO_START_IN_S if timeout is None else timeout

        start_time = time()
        schedulers = []
        while (len(schedulers) == 0
               and (time() - start_time) <= timeout):
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

        assert len(schedulers) == 1, 'Multiple schedulers are found (!?)'  # TODO: Is this possible?

        scheduler_uri = schedulers.get(f'Scheduler_{self.run_id}')
        self.logger.info(f'Scheduler located. Uri is {scheduler_uri}.')

        return scheduler_uri

    def __setup_logger(self, level=None):
        # TODO: write this to the hpobenchexp utils.
        logger = logging.getLogger(f'{self.worker_id}')
        level = level or logging.root.level

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_stream = logging.StreamHandler()
        console_stream.setLevel(logging.DEBUG)
        console_stream.setFormatter(formatter)

        logger.addHandler(console_stream)
        logger.setLevel(level)

        return logger

    def __del__(self):
        self.logger.debug('Calling the Shutdown method')
        self.is_alive()
        self.logger.debug('Stop the daemon.')
        self.is_running = False


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--credentials_dir', type=str, required=True,
                        help='The address of the nameserver is stored here')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Unique name of the run')
    parser.add_argument('--worker_id', type=int, required=True,
                        help='Unique name of the worker')
    return parser.parse_args()
