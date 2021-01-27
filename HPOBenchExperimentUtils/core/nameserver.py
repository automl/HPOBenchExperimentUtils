import Pyro4.naming
import logging
from pathlib import Path
import json
import threading

logger = logging.getLogger('Nameserver')


def start_nameserver(ns_ip: str = '127.0.0.1',
                     ns_port: int = 0,
                     credentials_file: Path = Path.cwd() / f'HPBenchExpUtils_pyro4_nameserver_0.json',
                     thread_name: str = 'HPOBenchExpUtils'):
    """ ns_port = 0 means a random port """

    # Let the nameserver clear its registrations automatically  every X seconds
    from Pyro4.configuration import config
    config.NS_AUTOCLEAN = 30

    try:
        uri, ns, _ = Pyro4.naming.startNS(host=ns_ip, port=ns_port)
    except OSError as e:
        logger.warning('Nameserver is already in use.')
        raise e

    ns_ip, ns_port = uri.location.split(':')
    ns_port = int(ns_port)

    # save credentials to file
    with credentials_file.open('w') as fh:
        json.dump([ns_ip, ns_port], fh)
    logger.debug(f'The credentials file is here: {credentials_file}')

    thread = threading.Thread(target=ns.requestLoop, name=thread_name, daemon=True)
    thread.start()
    logger.info(f'The nameserver is running on {ns_ip}:{ns_port}')

    return ns_ip, ns_port
