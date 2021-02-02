import logging
import shutil
import sys
from collections import namedtuple
from pathlib import Path
from time import sleep, time
from typing import Dict, Union, Tuple, List, NamedTuple

import Pyro4
import Pyro4.errors
import Pyro4.naming
import Pyro4.util
from oslo_concurrency import lockutils

from HPOBenchExperimentUtils.core import SCHEDULER_PING_WORKERS_INTERVAL_IN_S, SCHEDULER_TIMEOUT_WORKER_DISCOVERY_IN_S
from HPOBenchExperimentUtils.core.daemon_object import DaemonObject
from HPOBenchExperimentUtils.utils import VALIDATED_RUNHISTORY_FILENAME
from HPOBenchExperimentUtils.utils.io import write_line_to_file


# Organize the tasks of the scheduler in a list with objects of the following form:
Content = namedtuple('Content', ['Configuration', 'Fidelity', 'Additional'])

sys.excepthook = Pyro4.util.excepthook


@Pyro4.behavior(instance_mode='single')
class Scheduler(DaemonObject):
    def __init__(self, run_id: Union[int, str], ns_ip: str, ns_port: int, output_dir: Path,
                 contents: List[NamedTuple], debug: bool = False):
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Unique name for the run
        self.run_id = run_id

        # Unique name for the scheduler
        self.scheduler_id = f'Scheduler_{run_id}'

        self.output_dir = Path(output_dir)
        self.runhistory_file = self.output_dir / VALIDATED_RUNHISTORY_FILENAME

        self.credentials_file = self.output_dir / f'HPBenchExpUtils_pyro4_nameserver_{run_id}.json'

        self.lock_dir = self.output_dir / f'Lockfiles_{self.scheduler_id}'
        try:
            self.lock_dir.mkdir(parents=True)
        except FileExistsError:
            self.logger.warning('The lock files still exist. Make sure that this process does not run twice.')

        self.contents = contents  # type: List[NamedTuple]
        self.total_num_contents = len(self.contents)

        # If the scheduler has not seen for defined time period a worker, we stop the scheduler, because we assume that
        # there is an error.
        self.time_since_last_connection_to_worker = time()

        # This mapping keeps track of the (configurations, fidelity)-pair, which are currently processed a worker, which
        # is represented by its URI.
        self.configuration_fidelity_to_uri_mapping = {}
        self.uri_to_configuration_fidelity_mapping = {}

        # This dictionary stores the results. The key is a pair of configuration and fidelity.
        self.results = {}

        # Start the pyro daemon in the background, such that the object can be found by the worker.
        super(Scheduler, self).__init__(ns_ip=ns_ip, ns_port=ns_port, registration_name=self.scheduler_id,
                                        logger=self.logger)

    def run(self):
        """
        Main loop of the scheduler.
        Prints automatically the number of configurations and pings the worker to see if they are alive.

        If we haven't seen a worker in the last SCHEDULER_TIMEOUT_WORKER_DISCOVERY_IN_S timesteps, we have to assume that only the
        scheduler is alive, so we shut the scheduler down. Another reason could be that server and workers can't find
        each other. But then, we should kill the run anyway.
        """
        self.time_since_last_connection_to_worker = time()

        # Loop until we have no configs to validate left and no configuration is currrently processed.
        while len(self.contents) != 0 or len(self.uri_to_configuration_fidelity_mapping) != 0:

            self.logger.debug(f'Still {len(self.contents)}|{self.total_num_contents} in queue. '
                              f'Currently {len(self.uri_to_configuration_fidelity_mapping)} configs are processed by '
                              'workers.')

            # Ping the workers and check if they are still responsive.
            self.check_worker()

            # Check the time limit
            if time() - self.time_since_last_connection_to_worker > SCHEDULER_TIMEOUT_WORKER_DISCOVERY_IN_S:
                self.logger.error(f'We could not find a single worker during the last {SCHEDULER_TIMEOUT_WORKER_DISCOVERY_IN_S} '
                                  'seconds. Please check if the workers are able to find the Pyro4.nameserver.'
                                  'We kill the scheduler process for now.')
                break

            sleep(SCHEDULER_PING_WORKERS_INTERVAL_IN_S)

        self.logger.info('Going to shutdown the scheduler.')

    def check_worker(self):
        """
        Look up all the workers registered in the name server and ping them.
        If a worker is not responsive or in a failure state, reschedule its current configuration.

        Note that the nameserver is configured to automatically clear broken pyro objects.
        """
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
                        # Check if all "currently working" worker are still available
                        if worker_uri in self.uri_to_configuration_fidelity_mapping:
                            self.logger.warning('A worker which has not registered its results, is not '
                                                'reachable anymore. Reschedule its configuration.')

                            config, fidelity, additional = self._get_entry_from_mapping_by_uri(worker_uri)
                            self.contents.append(Content(config, fidelity, additional))
                            self._remove_from_mapping_by_uri(worker_uri)

                        self.logger.error(f'Worker {worker_uri} is not reachable. Remove it from the nameserver')

                        try:
                            worker._pyroRelease()
                            worker.is_running = False
                        except Pyro4.errors.CommunicationError:
                            pass

    @Pyro4.expose
    def get_content(self, caller_uri: Pyro4.core.URI) -> Tuple[Dict, Dict, Dict, bool]:

        configuration, fidelity, additional, msg_contains_item = {}, {}, {}, False

        self.logger.debug('Acquire Lock for getting a task')
        with lockutils.lock(name='get_content_lock', external=True, do_log=False, delay=0.3,
                                           lock_path=f'{self.lock_dir}/lock_get_content'):
            if len(self.contents) >= 1:
                configuration, fidelity, additional = self.contents.pop(0)

                self._add_to_mapping(uri=caller_uri,
                                     configuration=configuration, fidelity=fidelity, additional=additional)

                msg_contains_item = True
            else:
                self.logger.info(f'GetContent: No items left')

        self.logger.info(f'GetContent: {configuration}')
        return configuration, fidelity, additional, msg_contains_item

    @Pyro4.expose
    def register_result(self, configuration: Dict, fidelity: Dict, additional: Dict, result_dict: Dict) -> bool:
        self.logger.debug(f'Register Result for configuration {configuration} and fidelity {fidelity} '
                          f'- Additional: {additional}')
        self.logger.debug(f'Received Result: {result_dict}')

        self.results[(str(configuration), str(fidelity), str(additional))] = result_dict
        self._remove_from_mapping_by_config_fidelity(configuration, fidelity, additional)

        with lockutils.lock(name='write_results_to_file', external=True, do_log=False, delay=0.3,
                                         lock_path=f'{self.lock_dir}/write_results'):
            write_line_to_file(self.runhistory_file, result_dict)

        return True

    def get_results_by_configuration(self):
        results_without_fidelity_key = {config: value for (config, fidelity, additional), value in self.results.items()}
        return results_without_fidelity_key

    def _get_entry_from_mapping_by_uri(self, uri: Union[str, Pyro4.core.URI]):
        uri = self.__cast_uri_to_str(uri)
        return self.uri_to_configuration_fidelity_mapping[uri]

    def _get_entry_from_mapping_by_config_fidelity(self, configuration, fidelity, additional):
        return self.configuration_fidelity_to_uri_mapping[(str(configuration, str(fidelity), str(additional)))]

    def _add_to_mapping(self, uri, configuration, fidelity, additional):
        uri = self.__cast_uri_to_str(uri)
        self.configuration_fidelity_to_uri_mapping[(str(configuration), str(fidelity), str(additional))] = uri
        self.uri_to_configuration_fidelity_mapping[uri] = (configuration, fidelity, additional)

    def _remove_from_mapping_by_uri(self, uri):
        uri = self.__cast_uri_to_str(uri)
        configuration, fidelity, additional = self.uri_to_configuration_fidelity_mapping[uri]
        self.__remove_from_mapping_by_config_fidelity(uri, configuration, fidelity, additional)

    def _remove_from_mapping_by_config_fidelity(self, configuration, fidelity, additional):
        uri = self.configuration_fidelity_to_uri_mapping[(str(configuration), str(fidelity), str(additional))]
        self.__remove_from_mapping_by_config_fidelity(uri, configuration, fidelity, additional)

    def __remove_from_mapping_by_config_fidelity(self, uri, configuration, fidelity, additional):
        uri = self.__cast_uri_to_str(uri)
        self.logger.debug(f'Remove entry from mapping: {uri} {configuration} {fidelity} {additional}')
        del self.configuration_fidelity_to_uri_mapping[(str(configuration), str(fidelity), str(additional))]
        del self.uri_to_configuration_fidelity_mapping[uri]

    @staticmethod
    def __cast_uri_to_str(uri: Union[str, Pyro4.core.URI]):
        if isinstance(uri, Pyro4.core.URI):
            uri = uri.asString()
        return uri

    def __del__(self):
        self.logger.debug('Shutdown..')

        if self.credentials_file.exists():
            self.logger.debug(f'Going to delete the credentials file: {self.credentials_file}')
            self.credentials_file.unlink()

        if self.lock_dir.exists():
            self.logger.debug(f'Going to delete the lock file directory {self.lock_dir}')
            shutil.rmtree(self.lock_dir)

        super(Scheduler, self).__del__()
