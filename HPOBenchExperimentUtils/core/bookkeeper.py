import logging
from concurrent.futures import TimeoutError
from functools import wraps
from pathlib import Path
from time import time
from typing import Union, List, Dict, Any

import ConfigSpace as CS
import numpy as np
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from pebble import concurrent
from oslo_concurrency import lockutils

from HPOBenchExperimentUtils.core.record import Record
from HPOBenchExperimentUtils.utils import MAXINT, RUNHISTORY_FILENAME, TRAJECTORY_V1_FILENAME, \
    VALIDATED_RUNHISTORY_FILENAME
from HPOBenchExperimentUtils.utils.io import write_line_to_file

logger = logging.getLogger(__name__)


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

            self.increase_total_tae_used(1)

            start_time = time()
            result_dict = function(self, configuration, fidelity, rng, **kwargs)

            # Throw an time error if the function evaluation takes more time than the specified cutoff value.
            # Note: This check is intended to
            try:
                result_dict = result_dict.result()
            except TimeoutError:
                # TODO: Fidelity is a dict here. How to extract the value? What to do if it is None?
                f = _safe_cast_config(fidelity) or None
                self.increase_total_fuel_used(list(f.values())[0] or 0)
                self.increase_total_time_used(self.cutoff_limit_in_s)

                record = Record(function_value=MAXINT,
                                cost=self.cutoff_limit_in_s,
                                info={'fidelity': fidelity or -1234})

                return record.get_dictionary()

            # We can only compute the finish time after we obtain the result()
            finish_time = time()

            if not np.isfinite(result_dict["function_value"]):
                result_dict["function_value"] = MAXINT

            self.total_objective_costs += result_dict['cost']

            # Measure the total time since the start up. This is the budget which is used by the optimization procedure.
            # In case of a surrogate benchmark also take the "surrogate" costs into account.
            total_time_used = time() - self.boot_time
            if self.is_surrogate:
                total_time_used -= (finish_time - start_time)
                total_time_used += self.total_objective_costs

            # Time used for this configuration. The benchmark returns as cost the time of the function call +
            # the cost of the configuration. If the benchmark is a surrogate, the cost field includes the costs for the
            # function call, as well as surrogate costs. Thus, it is sufficient to use the costs returned by the
            # benchmark.
            time_for_evaluation = result_dict['cost']

            # if the fidelity is none: load it from the result dictionary.
            fidelity = _safe_cast_config(fidelity or result_dict['info']['fidelity'])
            configuration = _safe_cast_config(configuration)

            # Note: We update the proxy variable after checking the conditions here.
            #       This is because, we want to make sure, that this process is not be killed from outside
            #       before it was able to write the current result into the result file.
            if not total_time_exceeds_limit(total_time_used, self.wall_clock_limit_in_s, time()) \
                and not used_fuel_exceeds_limit(self.get_total_fuel_used(), self.fuel_limit) \
                and not tae_exceeds_limit(self.get_total_tae_used(), self.tae_limit) \
                    and not time_per_config_exceeds_limit(time_for_evaluation, self.cutoff_limit_in_s):

                record = Record(start_time=start_time,
                                finish_time=finish_time,
                                function_value=result_dict['function_value'],
                                fidelity=fidelity,
                                cost=result_dict['cost'],
                                configuration=configuration,
                                info=result_dict['info'],
                                function_call=self.get_total_tae_used(),
                                total_time_used=total_time_used,
                                total_objective_costs=self.total_objective_costs,
                                total_fuel_used=self.get_total_fuel_used())

                record = record.get_dictionary()

                log_file = self.log_file if not validate else self.validate_log_file
                self.write_line_to_file(log_file, record)

                if not validate:
                    self.calculate_incumbent(record)
            else:
                logger.info('We have reached a time limit. We do not write the current record into the trajectory.')

            self.set_total_time_used(total_time_used)
            self.increase_total_fuel_used(list(fidelity.values())[0] or 0)

            return result_dict
        return wrapped
    return wrapper


class Bookkeeper:
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 output_dir: Path,
                 total_time_proxy: Any,
                 total_tae_calls_proxy: Any,
                 total_fuel_used_proxy: Any,
                 wall_clock_limit_in_s: Union[int, float, None],
                 tae_limit: Union[int, None],
                 fuel_limit: Union[int, float, None],
                 cutoff_limit_in_s: Union[int, float],
                 is_surrogate: bool,
                 validate: bool = False):

        self.benchmark = benchmark
        self.log_file = output_dir / RUNHISTORY_FILENAME
        self.trajectory = output_dir / TRAJECTORY_V1_FILENAME
        self.validate_log_file = output_dir / VALIDATED_RUNHISTORY_FILENAME

        self.boot_time = time()
        self.total_objective_costs = 0

        self.inc_budget = None
        self.inc_value = None

        # This variable is a shared variable. A proxy to check from outside. It represents the time already used
        # by this benchmark. It also takes into account if the benchmark is a surrogate.
        self.total_time_proxy = total_time_proxy

        # Same as total time proxy, but counts the total used budget
        self.total_fuel_used_proxy = total_fuel_used_proxy

        # Sums up the number of target algorithm executions.
        self.total_tae_calls_proxy = total_tae_calls_proxy

        self.lock_dir = output_dir / 'lock_dir'
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_dir / 'attribute_lock'

        self.wall_clock_limit_in_s = wall_clock_limit_in_s
        self.tae_limit = tae_limit
        self.fuel_limit = fuel_limit
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

    def get_total_time_used(self):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            value = self.total_time_proxy.value
        return value

    def get_total_tae_used(self):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            value = self.total_tae_calls_proxy.value
        return value

    def get_total_fuel_used(self):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            value = self.total_fuel_used_proxy.value
        return value

    def set_total_time_used(self, total_time_used: float):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            self.total_time_proxy.value = total_time_used

    def set_total_tae_used(self, total_tae_used: float):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            self.total_tae_calls_proxy.value = total_tae_used

    def set_total_fuel_used(self, total_fuel_used_proxy: float):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            self.total_fuel_used_proxy.value = total_fuel_used_proxy

    def increase_total_time_used(self, total_time_used_delta: float):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            self.total_time_proxy.value = self.total_time_proxy.value + total_time_used_delta

    def increase_total_tae_used(self, total_tae_used: float):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            self.total_tae_calls_proxy.value = self.total_tae_calls_proxy.value + total_tae_used

    def increase_total_fuel_used(self, total_fuel_used_proxy: float):
        lock = lockutils.lock(name='attribute_lock', external=True, do_log=False,
                              lock_path=f'{self.lock_file}', delay=0.3)
        with lock:
            self.total_fuel_used_proxy.value = self.total_fuel_used_proxy.value + total_fuel_used_proxy

    @staticmethod
    def write_line_to_file(file, dict_to_store, mode='a+'):
        write_line_to_file(file, dict_to_store, mode)

    def calculate_incumbent(self, record: Dict):
        """
        Given a new record (challenging configuration), compare it to the current incumbent.
        If the challenger is either on a higher budget or on the same budget but with a better performance, make the
        challenger the new incumbent.

        If the challenger is better, then append it to the trajectory.

        Parameters
        ----------
        record : Dict
        """
        # If any progress has made, Bigger is better, etc.
        fidelity = list(record['fidelity'].values())[0]

        if self.inc_value is None \
            or (abs(fidelity - self.inc_budget) <= 1e-8 and ((self.inc_value - record['function_value']) > 1e-8)) \
            or (fidelity - self.inc_budget) > 1e-8:

            self.inc_value = record['function_value']
            self.inc_budget = fidelity
            self.write_line_to_file(self.trajectory, record, mode='a')

    def __del__(self):
        if self.lock_dir.exists():
            import shutil
            shutil.rmtree(self.lock_dir)

        self.benchmark.__del__()


def total_time_exceeds_limit(total_time_proxy, time_limit_in_s, start_time):
    # Wait for the optimizer to finish.
    # But in case the optimizer crashes somehow, also test for the real time here.
    # if the limit is None, this condition is not active.
    return (time_limit_in_s is not None
            and (total_time_proxy > time_limit_in_s
                 or time() - start_time > time_limit_in_s))


def used_fuel_exceeds_limit(total_fuel_used_proxy, max_fuel):
    return max_fuel is not None and total_fuel_used_proxy > max_fuel


def tae_exceeds_limit(total_tae_calls_proxy, max_tae_calls):
    return max_tae_calls is not None and total_tae_calls_proxy > max_tae_calls


def time_per_config_exceeds_limit(time_per_config, cutoff_limit):
    return time_per_config > cutoff_limit
