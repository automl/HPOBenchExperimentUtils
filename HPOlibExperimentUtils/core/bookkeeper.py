import logging
import os
from concurrent.futures import TimeoutError
from functools import wraps
from multiprocessing import Lock
from pathlib import Path
from time import time
from typing import Union, List, Dict, Any

import ConfigSpace as CS
import json_tricks
import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient
from pebble import concurrent

from HPOlibExperimentUtils.utils import MAXINT

logger = logging.getLogger('Bookkeeper')


def _get_dict_types(d):
    assert isinstance(d, Dict), "Expected to display items types for a dictionary, but received object of type %s" % \
                                type(d)
    return {k: type(v) if not isinstance(v, Dict) else _get_dict_types(v) for k, v in d.items()}


def _safe_cast_config(configuration):
    if isinstance(configuration, CS.Configuration):
        configuration = configuration.get_dictionary()
    if isinstance(configuration, np.ndarray):
        configuration = configuration.tolist()
    return configuration


def keep_track(validate=False):
    def wrapper(function):
        @wraps(function)
        def wrapped(self, configuration: Union[np.ndarray, List, CS.Configuration, Dict],
                    fidelity: Union[CS.Configuration, Dict, None] = None,
                    rng: Union[np.random.RandomState, int, None] = None, **kwargs):

            self.function_calls += 1
            start_time = time()
            result_dict = function(self, configuration, fidelity, rng, **kwargs)
            finish_time = time()

            # Throw an time error if the function evaluation takes more time than the specified cutoff value.
            # Note: This check is intended to
            try:
                result_dict = result_dict.result()
            except TimeoutError:
                self.set_total_time_used(self.cutoff_limit_in_s)
                return {'function_value': MAXINT,
                        'cost': self.cutoff_limit_in_s,
                        'info': {'fidelity': fidelity or -1234}}

            self.total_objective_costs += result_dict['cost']

            # Measure the total time since the start up. This is the budget which is used by the optimization procedure.
            # In case of a surrogate benchmark also take the "surrogate" costs into account.
            total_time_used = time() - self.boot_time
            if self.is_surrogate:
                total_time_used -= (finish_time - start_time)
                total_time_used += self.total_objective_costs

            # Time used for this configuration. The benchamrk returns as cost the time of the function call +
            # the cost of the configuration. If the benchmark is a surrogate, the cost field includes the costs for the
            # function call, as well as surrogate costs. Thus, it is sufficient to use the costs returned by the
            # benchmark.
            time_for_evaluation = result_dict['cost']

            # Note: We update the proxy variable after checking the conditions here.
            #       This is because, we want to make sure, that this process is not be killed from outside
            #       before it was able to write the current result into the result file.
            if (self.wall_clock_limit_in_s is None or total_time_used <= self.wall_clock_limit_in_s) \
                    and time_for_evaluation <= self.cutoff_limit_in_s:
                configuration = _safe_cast_config(configuration)
                # if the fidelity is none: load it from the result dictionary.
                fidelity = _safe_cast_config(fidelity or result_dict['info']['fidelity'])

                record = {'start_time': start_time,
                          'finish_time': finish_time,
                          'function_value': result_dict['function_value'],
                          'fidelity': fidelity,
                          'cost': result_dict['cost'],
                          'configuration': configuration,
                          'info': result_dict['info'],
                          'function_call': self.function_calls,
                          'total_time_used': total_time_used,
                          'total_objective_costs': self.total_objective_costs
                          }

                log_file = self.log_file if not validate else self.validate_log_file
                self.write_line_to_file(log_file, record)

                if not validate:
                    self.calculate_incumbent(record)

            self.set_total_time_used(total_time_used)
            return result_dict

        return wrapped
    return wrapper


class Bookkeeper:
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 output_dir: Path,
                 total_time_proxy: Any,
                 wall_clock_limit_in_s: Union[int, float, None],
                 cutoff_limit_in_s: Union[int, float],
                 is_surrogate: bool,
                 validate: bool = False):

        self.benchmark = benchmark
        self.log_file = output_dir / 'hpolib_runhistory.txt'
        self.trajectory = output_dir / 'hpolib_trajectory.txt'
        self.validate_log_file = output_dir / 'hpolib_runhistory_validation.txt'
        # self.validate_trajectory = output_dir / 'hpolib_trajectory_validation.txt'
        # self.validate_log_db = output_dir / 'hpolib_runhistory_validation.db'

        self.boot_time = time()
        self.total_objective_costs = 0
        # TODO: Create proxy for tae count
        self.function_calls = 0

        self.inc_budget = None
        self.inc_value = None

        # self.inc_budget_validated = None
        # self.inc_value_validated = None

        # This variable is a share variable. A proxy to check from outside. It represents the time already used for this
        # benchmark. It also takes into account if the benchmark is a surrogate.
        self.total_time_proxy = total_time_proxy
        self.global_time_lock = Lock()

        self.wall_clock_limit_in_s = wall_clock_limit_in_s
        self.cutoff_limit_in_s = cutoff_limit_in_s
        self.is_surrogate = is_surrogate

        if not validate:
            self.write_line_to_file(self.log_file, {'boot_time': self.boot_time}, mode='w')
            self.write_line_to_file(self.trajectory, {'boot_time': self.boot_time}, mode='w')
        else:
            if self.validate_log_file.exists():
                logger.warning(f'The validation log file already exists. The results will be appended.')
            self.write_line_to_file(self.validate_log_file, {'boot_time': self.boot_time}, mode='a+')

    @keep_track(validate=False)
    def objective_function(self, configuration: Union[np.ndarray, List, CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        @concurrent.process(timeout=self.cutoff_limit_in_s)
        def __objective_function(configuration, fidelity, **benchmark_settings_for_sending):
            return self.benchmark.objective_function(configuration=configuration,
                                                     fidelity=fidelity,
                                                     **benchmark_settings_for_sending)

        result_dict = __objective_function(configuration=configuration,
                                           fidelity=fidelity,
                                           rng=rng,
                                           **kwargs)
        return result_dict

    @keep_track(validate=True)
    def objective_function_test(self, configuration: Union[np.ndarray, List, CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        @concurrent.process(timeout=self.cutoff_limit_in_s)
        def __objective_function_test(configuration, fidelity, **benchmark_settings_for_sending):
            return self.benchmark.objective_function_test(configuration=configuration,
                                                          fidelity=fidelity,
                                                          **benchmark_settings_for_sending)

        result_dict = __objective_function_test(configuration=configuration,
                                                fidelity=fidelity,
                                                rng=rng,
                                                **kwargs)
        return result_dict

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchmark.get_configuration_space(seed)

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchmark.get_fidelity_space(seed)

    def get_meta_information(self) -> Dict:
        return self.benchmark.get_meta_information()

    def set_total_time_used(self, total_time_used: float):
        with self.global_time_lock:
            self.total_time_proxy.value = total_time_used

    def get_total_time_used(self):
        with self.global_time_lock:
            return self.total_time_proxy.value

    @staticmethod
    def write_line_to_file(file, dict_to_store, mode='a+'):
        with file.open(mode) as fh:
            try:
                json_tricks.dump(dict_to_store, fh)
            except TypeError as e:
                logger.error(f"Failed to serialize dictionary to JSON. Received the following types as "
                             f"input:\n{_get_dict_types(dict_to_store)}")
                raise e
            fh.write(os.linesep)

    def calculate_incumbent(self, record: Dict):
        # If any progress has made, Bigger is better, etc.
        fidelity = list(record['fidelity'].values())[0]

        if self.inc_value is None \
            or (abs(fidelity - self.inc_budget) <= 1e-8 and ((self.inc_value - record['function_value']) > 1e-8)) \
            or (fidelity - self.inc_budget) > 1e-8:

            self.inc_value = record['function_value']
            self.inc_budget = fidelity
            self.write_line_to_file(self.trajectory, record, mode='a')

    def __del__(self):
        self.benchmark.__del__()
