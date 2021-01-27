import Pyro4
import Pyro4.errors

import threading
from time import sleep


class DaemonObject(object):
    def __init__(self, ns_ip, ns_port, registration_name, logger):
        # This flag indicates that the daemon is running, so the worker can be found by the scheduler.
        # The starting procedure sets this flag to True.
        # When it is set to false, then the daemon automatically stops.
        self.thread_is_running = False

        self.logger = logger

        # Where to find the nameserver:
        self.ns_ip = ns_ip
        self.ns_port = ns_port

        # This object is registered in the nameserver with this name.
        self.registration_name = registration_name
        self.uri = None
        self.logger.debug('Going to start the daemon of this object in the background.')
        self.thread = self.__start_daemon()
        self.logger.info('Registered this object in the nameserver.')
        self.logger.debug(f'The object has the uri: {str(self.uri)}')

    def __start_daemon(self):
        # Start the Pyro daemon. The daemon allows other objects to interact with this worker.
        # Also: make it a daemon process. Then the thread gets killed when the worker is shut down
        thread = threading.Thread(target=self.__run_daemon, name=f'Thread: {self.registration_name}', daemon=True)
        thread.start()

        # give the thread some time to start
        sleep(1)
        return thread

    def __run_daemon(self):
        """
        This is the background thread that registers the object in the nameserver. Also, it starts the request loop
        to make the object able to receive requests.

        When the flag `thread_is_running` is set to False (from outside), then the request loop stops and the daemon shuts
        down. Also, since this thread is a daemon thread, it automatically stops, when the main thread terminates.
        """
        self.thread_is_running = True

        try:
            with Pyro4.Daemon() as daemon:
                self.uri = daemon.register(self)

                with Pyro4.locateNS(host=self.ns_ip, port=self.ns_port) as nameserver:
                    nameserver.register(self.registration_name, self.uri)

                daemon.requestLoop(loopCondition=lambda: self.thread_is_running)
                self.logger.info(f'Stopped the request loop. This object is not reachable anymore')
        except Pyro4.errors.NamingError as e:
            self.logger.error('We could not find the nameserver. Please make sure that it is running.')
            self.logger.exception(e)
