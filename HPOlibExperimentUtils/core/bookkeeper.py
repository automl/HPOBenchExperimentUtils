import json
import os
from concurrent.futures import TimeoutError
from functools import wraps
from multiprocessing import Lock
from pathlib import Path
from time import time
from typing import Union, List, Dict, Any

import ConfigSpace as CS
import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient
from pebble import concurrent

from HPOlibExperimentUtils.utils import MAXINT


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
                        'info': {'fidelity': fidelity}}

            self.total_objective_costs += result_dict['cost']

            # Measure the total time since the start up. This is the budget which is used by the optimization procedure.
            # In case of a surrogate benchmark also take the "surrogate" costs into account.
            total_time_used = time() - self.boot_time
            if self.is_surrogate:
                total_time_used += self.total_objective_costs

            # Time used for this configuration. The benchamrk returns as cost the time of the function call +
            # the cost of the configuration. If the benchmark is a surrogate, the cost field includes the costs for the
            # function call, as well as surrogate costs. Thus, it is sufficient to use the costs returned by the
            # benchmark.
            time_for_evaluation = result_dict['cost']

            # Note: We update the proxy variable after checking the conditions here.
            #       This is because, we want to make sure, that this process is not be killed from outside
            #       before it was able to write the current result into the result file.
            if total_time_used <= self.wall_clock_limit_in_s \
                    and time_for_evaluation <= self.cutoff_limit_in_s:
                configuration = _safe_cast_config(configuration)
                fidelity = _safe_cast_config(fidelity)

                record = {'start_time': start_time,
                          'finish_time': finish_time,
                          'function_value': result_dict['function_value'],
                          'fidelity': fidelity,
                          'cost': result_dict['cost'],
                          'configuration': configuration,
                          'info': result_dict['info']
                          }

                log_file = self.log_file if not validate else self.validate_log_file
                self.write_line_to_file(log_file, record)
                self.calculate_incumbent(record)

            self.set_total_time_used(total_time_used)
            return result_dict

        return wrapped
    return wrapper


class Bookkeeper:
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 output_dir: Path,
                 total_time_proxy: Any,
                 wall_clock_limit_in_s: Union[int, float],
                 cutoff_limit_in_s: Union[int, float],
                 is_surrogate: bool,
                 validate: bool = False):

        self.benchmark = benchmark
        self.log_file = output_dir / 'hpolib_runhistory.txt'
        self.trajectory = output_dir / 'hpolib_trajectory.txt'
        self.validate_log_file = output_dir / 'hpolib_runhistory_validation.txt'

        self.boot_time = time()
        self.total_objective_costs = 0  # TODO: Create proxy for tae count

        self.inc_budget = None
        self.inc_value = None

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

        self.function_calls = 0

    @keep_track(validate=False)
    def objective_function(self, configuration: Union[np.ndarray, List, CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        @concurrent.process(timeout=self.cutoff_limit_in_s)
        def __objective_function(configuration, fidelity, **benchmark_settings_for_sending):
            result_dict = self.benchmark.objective_function(configuration=configuration,
                                                            fidelity=fidelity,
                                                            **benchmark_settings_for_sending)
            return result_dict

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
            result_dict = self.benchmark.objective_function(configuration=configuration,
                                                            fidelity=fidelity,
                                                            **benchmark_settings_for_sending)
            return result_dict

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
    def write_line_to_file(file, dict_to_store, mode='a'):
        with file.open(mode) as fh:
            json.dump(dict_to_store, fh)
            fh.write(os.linesep)

    def calculate_incumbent(self, record: Dict):
        # If any progress has made, Bigger is better, etc.
        fidelity = list(record['fidelity'].values())[0]

        if self.inc_value is None \
                or (abs(fidelity - self.inc_budget) <= 1e-8 and ((self.inc_value - record['function_value']) > 1e-8))\
                or (fidelity - self.inc_budget) > 1e-8:
            self.inc_value = record['function_value']
            self.inc_budget = fidelity

            self.write_line_to_file(self.trajectory, record, mode='a')