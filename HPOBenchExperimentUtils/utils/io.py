import logging
import json_tricks
import os

from typing import Dict


logger = logging.getLogger(__name__)


def write_line_to_file(file, dict_to_store, mode='a+'):
    with file.open(mode) as fh:
        try:
            json_tricks.dump(dict_to_store, fh)
        except TypeError as e:
            logger.error(f"Failed to serialize dictionary to JSON. Received the following types as "
                         f"input:\n{_get_dict_types(dict_to_store)}")
            raise e
        fh.write(os.linesep)


def _get_dict_types(d):
    assert isinstance(d, Dict), f"Expected to display items types for a dictionary, but received object of " \
                                f"type {type(d)}"
    return {k: type(v) if not isinstance(v, Dict) else _get_dict_types(v) for k, v in d.items()}
