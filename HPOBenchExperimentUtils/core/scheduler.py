import Pyro4
import Pyro4.naming
import Pyro4.errors
import Pyro4.util
import threading
import json
import random
import shutil
import sys

from pathlib import Path
from time import sleep, time
from typing import Dict, Any, Union, Tuple

from oslo_concurrency import lockutils
from HPOBenchExperimentUtils.core.DaemonObject import DaemonObject

import logging

logging.basicConfig(level=logging.DEBUG)

main_logger = logging.getLogger('Main Process')

# TODO: Define them better
PING_INTERVAL_IN_S = 5
TIMEOUT_NO_WORKER_IN_S = 120

sys.excepthook = Pyro4.util.excepthook


@Pyro4.behavior(instance_mode='single')
class Scheduler(DaemonObject):
    def __init__(self, run_id: Union[int, str], ns_ip: str, ns_port: int, output_dir: Path):
        self.logger = logging.getLogger('Scheduler')

        self.run_id = run_id

        # Unique name for the scheduler
        self.scheduler_id = f'Scheduler_{run_id}'

        self.output_dir = Path(output_dir)
        self.lock_dir = self.output_dir / f'Lockfiles_{self.scheduler_id}'
        self.lock_dir.mkdir(parents=True)

        self.contents = [f'config_{i}' for i in range(30)]
        self.total_num_contents = len(self.contents)
        self.finished = False

        # If the scheduler has not seen for defined time period a worker, we stop the scheduler, because we assume that
        # there is an error.
        self.time_since_last_connection_to_worker = time()

        # This mapping keeps track of the (configurations, fidelity)-pair, which are currently processed.
        self.configuration_fidelity_to_uri_mapping = {}
        self.uri_to_configuration_fidelity_mapping = {}

        # The final results
        self.results = {}

        # Start the pyro deamon in the background, such that the object can be found by the worker.
        super(Scheduler, self).__init__(ns_ip=ns_ip, ns_port=ns_port,
                                        registration_name=self.scheduler_id, logger=self.logger)

    def run(self):
        self.time_since_last_connection_to_worker = time()
        while len(self.contents) != 0 or len(self.uri_to_configuration_fidelity_mapping) != 0:

            self.logger.debug(f'Still {len(self.contents)}|{self.total_num_contents} in queue. '
                              f'Currently {len(self.uri_to_configuration_fidelity_mapping)} configs are processed by '
                              'workers.')

            # Ping the workers and check if they are still responsive.
            self.check_worker()

            # If we haven't see a worker in the last X timesteps, we have to assume that only the scheduler is still
            # so we shut it down. It could also happen that server and workers can't find each other. But then we
            # should kill the run anyway.
            if time() - self.time_since_last_connection_to_worker > TIMEOUT_NO_WORKER_IN_S:
                self.logger.error(f'We could not find a single worker during the last {TIMEOUT_NO_WORKER_IN_S} '
                                  'seconds. Please check if the workers are able to find the Pyro4.nameserver.'
                                  'We kill the scheduler process for now.')
                break

            sleep(PING_INTERVAL_IN_S)

        self.logger.info('Going to shutdown the scheduler.')

    @Pyro4.expose
    def get_content(self, caller_uri) -> Tuple[Dict, Dict, bool]:

        configuration, fidelity, msg_contains_item = {}, {}, False
        lock = lockutils.lock(name='get_content_lock', external=True,
                              lock_path=f'{self.lock_dir}/lock_get_content', delay=0.3)
        with lock:
            if len(self.contents) >= 1:
                logging.debug('Test Lock across Threads')
                sleep(random.random() + 1)
                configuration = self.contents.pop(0)
                fidelity = {}

                self._add_to_mapping(uri=caller_uri, configuration=configuration, fidelity=fidelity)

                msg_contains_item = True
            else:
                self.logger.info(f'GetContent: No items left')

        self.logger.info(f'GetContent: {configuration}')
        return configuration, {}, msg_contains_item

    @Pyro4.expose
    def register_result(self, configuration: Dict, fidelity: Dict, result_dict: Dict) -> bool:
        self.logger.debug(f'Register Result for configuration {configuration} and fidelity {fidelity}')
        self.logger.info(f'Received Result: {result_dict}')
        self.results[(str(configuration), str(fidelity))] = result_dict
        self._remove_from_mapping_by_config_fidelity(configuration, fidelity)

        # TODO: Write results to file/database?
        pass

        return True

    @Pyro4.expose
    def is_finished(self):
        return self.finished

    def check_worker(self):
        # dont check if a worker is available. The nameserver automatically clears the broken ones.
        # Only check if the worker is currently processing something.
        with Pyro4.locateNS(host=self.ns_ip, port=self.ns_port) as ns:
            workers = ns.list(prefix=f'Worker_{self.run_id}')

            if len(workers) != 0:
                self.time_since_last_connection_to_worker = time()

            # self.time_since_last_connection_to_worker = self.time_since_last_connection_to_worker or len(workers) > 0

            for worker_name, worker_uri in workers.items():
                with Pyro4.Proxy(worker_uri) as worker:
                    # Check if worker is alive
                    try:
                        worker._pyroBind()
                        worker.is_alive()
                    except Pyro4.errors.CommunicationError:
                        # TODO: check if all "currently working" worker are still available
                        if worker_uri in self.uri_to_configuration_fidelity_mapping.keys():
                            self.logger.warning('A worker which has not registered its results, is not '
                                                'reachable anymore. Reschedule its configuration.')

                            # TODO: is it possible that the worker has sent its result, but the message has a "delay",
                            #       so that the wokre is already shutdown and the result is still on the way?
                            #       maybe its better to make the register call not a one way function?!
                            pass
                            config, fidelity = self._get_entry_from_mapping_by_uri(worker_uri)
                            self._remove_from_mapping_by_config_fidelity(config, fidelity)
                            self.contents.append(config)  # TODO: and fidelity

                        self.logger.error('Worker is not reachable?')
                        try:
                            worker._pyroRelease()
                            worker.is_running = False
                        except Pyro4.errors.CommunicationError:
                            pass

    def _get_entry_from_mapping_by_uri(self, uri: Union[str, Pyro4.core.URI]):
        uri = self.__cast_uri_to_str(uri)
        return self.uri_to_configuration_fidelity_mapping[uri]

    def _get_entry_from_mapping_by_config_fidelity(self, configuration, fidelity):
        return self.configuration_fidelity_to_uri_mapping[(str(configuration, str(fidelity)))]

    def _add_to_mapping(self, uri, configuration, fidelity):
        uri = self.__cast_uri_to_str(uri)
        self.configuration_fidelity_to_uri_mapping[(str(configuration), str(fidelity))] = uri
        self.uri_to_configuration_fidelity_mapping[uri] = (configuration, fidelity)

    def _remove_from_mapping_by_uri(self, uri):
        uri = self.__cast_uri_to_str(uri)
        configuration, fidelity = self.uri_to_configuration_fidelity_mapping[uri]
        self.__remove_from_mapping_by_config_fidelity(uri, configuration, fidelity)

    def _remove_from_mapping_by_config_fidelity(self, configuration, fidelity):
        uri = self.configuration_fidelity_to_uri_mapping[(str(configuration), str(fidelity))]
        self.__remove_from_mapping_by_config_fidelity(uri, configuration, fidelity)

    def __remove_from_mapping_by_config_fidelity(self, uri, configuration, fidelity):
        uri = self.__cast_uri_to_str(uri)
        self.logger.debug(f'Remove entry from mapping: {uri} {configuration} {fidelity}')
        del self.configuration_fidelity_to_uri_mapping[(str(configuration), str(fidelity))]
        del self.uri_to_configuration_fidelity_mapping[uri]

    def __cast_uri_to_str(self, uri: Union[str, Pyro4.core.URI]):
        if isinstance(uri, Pyro4.core.URI):
            uri = uri.asString()
        return uri

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        self.logger.debug('Shutdown..')
        if self.lock_dir.exists():
            self.logger.debug(f'Going to delete the lock file directory {self.lock_dir}')
            shutil.rmtree(self.lock_dir)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, required=True,
                        help='The output as well as the credentials are here stored.')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Unique name of the run')
    parser.add_argument('--interface', type=str, help='The network interface name, e.g. lo or localhost')
    args = parser.parse_args()
    return args


