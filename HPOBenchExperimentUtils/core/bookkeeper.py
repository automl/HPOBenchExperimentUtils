import json

import logging
import shutil
import copy
from concurrent.futures import TimeoutError
from functools import wraps
from pathlib import Path
from time import time
from typing import Union, List, Dict, Any

import ConfigSpace as CS
import numpy as np
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
        def wrapped(self, configuration: Union[CS.Configuration, Dict],
                    fidelity: Union[CS.Configuration, Dict, None] = None,
                    rng: Union[np.random.RandomState, int, None] = None, **kwargs):
            start_time = time()
            result_dict = function(self, configuration, fidelity, rng, **kwargs)

            # Throw an time error if the function evaluation takes more time than the specified cutoff value.
            # Note: This check is intended to
            try:
                result_dict = result_dict.result()
            except TimeoutError:
                # TODO: Fidelity is a dict here. How to extract the value? What to do if it is None?
                f = _safe_cast_config(fidelity) or None

                self.add_and_write_resource(total_time_delta=self.cutoff_limit_in_s,
                                            total_tae_calls_delta=1,
                                            total_fuel_used_delta=list(f.values())[0] or 0,
                                            total_objective_costs_delta=self.cutoff_limit_in_s)

                record = Record(function_value=MAXINT,
                                cost=self.cutoff_limit_in_s,
                                info={'fidelity': fidelity or -1234})

                return record.get_dictionary()

            # We can only compute the finish time after we obtain the result()
            finish_time = time()

            if not np.isfinite(result_dict["function_value"]):
                result_dict["function_value"] = MAXINT

            # if the fidelity is none: load it from the result dictionary.
            fidelity = _safe_cast_config(fidelity or result_dict['info']['fidelity'])
            configuration = _safe_cast_config(configuration)

            resource_lock = lockutils.lock(name=self.resource_lock_file, external=True, do_log=False,
                                           lock_path=str(self.lock_dir), delay=0.01)
            with resource_lock:
                resources = self._load_resource_file_without_lock(self.resource_file)

                total_objective_costs = resources['total_objective_costs'] + result_dict['cost']

                # Measure the total time since the start up. This is the budget which is used by the optimization
                # procedure. In case of a surrogate benchmark also take the "surrogate" costs into account.
                total_time_used = time() - resources['start_time']
                if self.is_surrogate:
                    total_time_used -= (finish_time - start_time)
                    total_time_used += total_objective_costs

                # Time used for this configuration. The benchmark returns as cost the time of the function call +
                # the cost of the configuration. If the benchmark is a surrogate, the cost field includes the costs
                # for the function call, as well as surrogate costs. Thus, it is sufficient to use the costs returned
                # by the benchmark.
                time_for_evaluation = result_dict['cost']
                resources['total_fuel_used_proxy'] += list(fidelity.values())[0] or 0
                resources['total_time_proxy'] = total_time_used
                resources['total_objective_costs'] += result_dict['cost']
                resources['total_tae_calls_proxy'] += 1

                # Note: We update the proxy variable after checking the conditions here.
                #       This is because, we want to make sure, that this process is not be killed from outside
                #       before it was able to write the current result into the result file.
                if not total_time_exceeds_limit(resources['total_time_proxy'], self.wall_clock_limit_in_s, time()) \
                    and not used_fuel_exceeds_limit(resources['total_fuel_used_proxy'], self.fuel_limit) \
                    and not tae_exceeds_limit(resources['total_tae_calls_proxy'], self.tae_limit) \
                        and not time_per_config_exceeds_limit(time_for_evaluation, self.cutoff_limit_in_s):

                    record = Record(start_time=start_time,
                                    finish_time=finish_time,
                                    function_value=result_dict['function_value'],
                                    fidelity=fidelity,
                                    cost=result_dict['cost'],
                                    configuration=configuration,
                                    info=result_dict['info'],
                                    function_call=resources['total_tae_calls_proxy'],
                                    total_time_used=total_time_used,
                                    total_objective_costs=resources['total_objective_costs'],
                                    total_fuel_used=resources['total_fuel_used_proxy'])
                    record = record.get_dictionary()

                    self.write_line_to_file(self.log_file if not validate else self.validate_log_file, record)
                else:
                    logger.info('We have reached a time limit. We do not write the current record into the trajectory.')

                self._write_resource_file_without_lock(self.resource_file, **resources)

            return result_dict
        return wrapped
    return wrapper


class Bookkeeper:
    def __init__(self,
                 benchmark_partial: Any,
                 output_dir: Path,
                 resource_file_dir: Path,
                 wall_clock_limit_in_s: Union[int, float, None],
                 tae_limit: Union[int, None],
                 fuel_limit: Union[int, float, None],
                 cutoff_limit_in_s: Union[int, float],
                 is_surrogate: bool,
                 validate: bool = False):

        self.benchmark_partial = benchmark_partial
        self.log_file = output_dir / RUNHISTORY_FILENAME
        self.trajectory = output_dir / TRAJECTORY_V1_FILENAME
        self.validate_log_file = output_dir / VALIDATED_RUNHISTORY_FILENAME
        self.resource_file = resource_file_dir / 'used_resources.json'

        self.lock_dir = output_dir / 'lock_dir'
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.resource_lock_file = 'resource_lock'

        self.wall_clock_limit_in_s = wall_clock_limit_in_s
        self.tae_limit = tae_limit
        self.fuel_limit = fuel_limit
        self.cutoff_limit_in_s = cutoff_limit_in_s
        self.is_surrogate = is_surrogate
        self.validate = validate

        # This means that this is the first time a book keeper is started. Write the boot time into the runhist + traj.
        if not self.resource_file.exists():
            resources = Bookkeeper.load_resource_file(self.resource_file, self.lock_dir, self.resource_lock_file)
            if not validate:
                self.write_line_to_file(self.log_file, {'boot_time': resources['start_time']}, mode='w')
                self.write_line_to_file(self.trajectory, {'boot_time': resources['start_time']}, mode='w')
            else:
                if self.validate_log_file.exists():
                    logger.warning(f'The validation log file already exists. The results will be appended.')
                self.write_line_to_file(self.validate_log_file, {'boot_time': resources['start_time']}, mode='a+')

    @keep_track(validate=False)
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        @concurrent.process(timeout=self.cutoff_limit_in_s)
        def __objective_function(configuration, fidelity, **benchmark_settings_for_sending):
            send = copy.copy(benchmark_settings_for_sending)
            if "random_seed_name" in benchmark_settings_for_sending:
                tmp_rng = np.random.RandomState()
                new_idx = int(tmp_rng.choice(benchmark_settings_for_sending["random_seed_choice"]))
                send[benchmark_settings_for_sending["random_seed_name"]] = new_idx
                del send["random_seed_name"]
                del send["random_seed_choice"]

            if "for_test" in send:
                del send["for_test"]

            for k in send:
                if send[k] == "None":
                    send[k] = None

            benchmark = self.benchmark_partial()
            result_dict = benchmark.objective_function(configuration=configuration,
                                                       fidelity=fidelity,
                                                       **send)
            return result_dict

        result_dict = __objective_function(configuration=configuration,
                                           fidelity=fidelity,
                                           rng=rng,
                                           **kwargs)
        return result_dict

    @keep_track(validate=True)
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        @concurrent.process(timeout=self.cutoff_limit_in_s)
        def __objective_function_test(configuration, fidelity, **benchmark_settings_for_sending):
            send = copy.copy(benchmark_settings_for_sending)
            if "random_seed_name" in benchmark_settings_for_sending:
                tmp_rng = np.random.RandomState()
                new_idx = int(tmp_rng.choice(benchmark_settings_for_sending["random_seed_choice"]))
                send[benchmark_settings_for_sending["random_seed_name"]] = new_idx
                del send["random_seed_name"]
                del send["random_seed_choice"]

            if "for_test" in send:
                for k in send["for_test"]:
                    send[k] = send["for_test"][k]
                del send["for_test"]

            for k in send:
                if send[k] == "None": send[k] = None

            benchmark = self.benchmark_partial()
            result_dict = benchmark.objective_function_test(configuration=configuration,
                                                            fidelity=fidelity,
                                                            **send)
            return result_dict

        result_dict = __objective_function_test(configuration=configuration,
                                                fidelity=fidelity,
                                                rng=rng,
                                                **kwargs)
        return result_dict

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        benchmark = self.benchmark_partial()
        cs = benchmark.get_configuration_space(seed)
        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        benchmark = self.benchmark_partial()
        fs = benchmark.get_fidelity_space(seed)
        return fs

    def get_meta_information(self) -> Dict:
        benchmark = self.benchmark_partial()
        meta = benchmark.get_meta_information()
        return meta

    @staticmethod
    def _load_resource_file_without_lock(resource_file):
        if not resource_file.exists():
            resources = dict(total_time_proxy=0.0,
                        total_tae_calls_proxy=0,
                        total_fuel_used_proxy=0.0,
                        total_objective_costs=0.0,
                        start_time=time())
            Bookkeeper._write_resource_file_without_lock(resource_file, **resources)
            logger.debug('Created Resource File.')
            return resources

        with resource_file.open('r') as fh:
            return json.load(fh)

    @staticmethod
    def _write_resource_file_without_lock(resource_file,
                                          total_time_proxy, total_tae_calls_proxy,
                                          total_fuel_used_proxy, total_objective_costs,
                                          start_time):
        resources = dict(total_time_proxy=total_time_proxy,
                         total_tae_calls_proxy=total_tae_calls_proxy,
                         total_fuel_used_proxy=total_fuel_used_proxy,
                         total_objective_costs=total_objective_costs,
                         start_time=start_time)

        with resource_file.open('w') as fp:
            json.dump(obj=resources, fp=fp)

    @staticmethod
    def load_resource_file(resource_file, lock_dir, resource_lock_file):
        resource_lock = lockutils.lock(name=resource_lock_file, external=True, do_log=False,
                                       lock_path=str(lock_dir), delay=0.01)
        with resource_lock:
            resources = Bookkeeper._load_resource_file_without_lock(resource_file)

        return resources

    def add_and_write_resource(self, total_time_delta=0, total_tae_calls_delta=0,
                               total_fuel_used_delta=0, total_objective_costs_delta=0):

        resource_lock = lockutils.lock(name=self.resource_lock_file, external=True, do_log=False,
                                       lock_path=str(self.lock_dir), delay=0.01)
        with resource_lock:
            old_resources = self._load_resource_file_without_lock(self.resource_file)
            resources = dict(total_time_proxy=total_time_delta + old_resources['total_time_proxy'],
                             total_tae_calls_proxy=total_tae_calls_delta + old_resources['total_tae_calls_proxy'],
                             total_fuel_used_proxy=total_fuel_used_delta + old_resources['total_fuel_used_proxy'],
                             total_objective_costs=total_objective_costs_delta + old_resources['total_objective_costs'],
                             start_time=old_resources['start_time'])
            with self.resource_file.open('w') as fp:
                json.dump(obj=resources, fp=fp)

    @staticmethod
    def write_line_to_file(file, dict_to_store, mode='a+'):
        write_line_to_file(file, dict_to_store, mode)

    def final_shutdown(self):
        if (self.lock_dir / self.resource_lock_file).exists():
            (self.lock_dir / self.resource_lock_file).unlink()

        self.__del__()

    def __del__(self):
        if self.lock_dir.exists():
            is_empty = not any(self.lock_dir.iterdir())
            if is_empty:
                shutil.rmtree(self.lock_dir)


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
