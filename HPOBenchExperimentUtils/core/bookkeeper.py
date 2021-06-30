import logging
import copy
from concurrent.futures import TimeoutError
from pathlib import Path
from time import time
from typing import Union, List, Dict, Any

import ConfigSpace as CS
import numpy as np
from pebble import concurrent

from HPOBenchExperimentUtils.core.data_objects import Record
from HPOBenchExperimentUtils.utils import MAXINT, RUNHISTORY_FILENAME, VALIDATED_RUNHISTORY_FILENAME
from HPOBenchExperimentUtils.utils.io import write_line_to_file
from HPOBenchExperimentUtils.resource_manager import FileBasedResourceManager

logger = logging.getLogger(__name__)


def _safe_cast_config(configuration):
    if isinstance(configuration, CS.Configuration):
        configuration = configuration.get_dictionary()
    if isinstance(configuration, np.ndarray):
        configuration = configuration.tolist()
    return configuration


class Bookkeeper:
    def __init__(self,
                 benchmark_partial: Any,
                 resource_manager: FileBasedResourceManager,
                 output_dir: Path,
                 is_surrogate: bool):

        self.benchmark_partial = benchmark_partial
        self.resource_manager = resource_manager
        self.resource_manager.start()

        self.output_dir = output_dir

        self.fidelity_space = self.get_fidelity_space()

        self.is_surrogate = is_surrogate

        self.run_history = output_dir / RUNHISTORY_FILENAME
        self.validated_run_history = output_dir / VALIDATED_RUNHISTORY_FILENAME

    def keep_track(self,
                   future_result,
                   random_config_id: str,
                   configuration: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   validate=False,
                   rng: Union[np.random.RandomState, int, None] = None,
                   **kwargs: Dict) -> Dict:

        start_time = time()

        if fidelity is None:
            fidelity = self.fidelity_space.get_default_configuration().get_dictionary()

        configuration = _safe_cast_config(configuration)
        fidelity = _safe_cast_config(fidelity)
        used_fuel = self.__extract_fuel_from_fidelity(fidelity, **kwargs)

        try:
            # Throw an time error if the function evaluation takes more time than the specified cutoff value.
            result_dict = future_result(configuration=configuration, fidelity=fidelity, rng=rng, **kwargs)
            result_dict = result_dict.result()
        except TimeoutError:
            cutoff = self.resource_manager.limits.cutoff_limit_in_s

            self.resource_manager.increase_resources(time_used_delta=cutoff,
                                                     tae_calls_delta=1,
                                                     # TODO: when a time out occurs, add the maximal fuel usage or the
                                                     #       one from the fidelity? (if none: max fuel)
                                                     fuel_used_delta=used_fuel,
                                                     objective_costs_delta=cutoff,
                                                     time_used_for_objective_call_delta=time() - start_time)

            record = Record(function_value=MAXINT,
                            cost=self.resource_manager.limits.cutoff_limit_in_s,
                            fidelity=fidelity,
                            info={'state': 'TIMEOUT'})
            return record.get_dictionary()

        # We can only compute the finish time after we obtain the result()
        finish_time = time()

        if 'fidelity' in result_dict['info'] and isinstance(result_dict['info']['fidelity'], Dict):
            fidelity = result_dict['info']['fidelity']

        if not np.isfinite(result_dict["function_value"]):
            result_dict["function_value"] = MAXINT

        with self.resource_manager.get_lock():
            resources = self.resource_manager.get_used_resources_without_lock()
            total_objective_costs = resources.total_objective_costs + result_dict['cost']
            total_time_used_for_objective_calls_in_s = resources.total_time_used_for_objective_calls_in_s
            total_time_used_for_objective_calls_in_s += (finish_time - start_time)

            # Measure the total time since the start up.
            # `Total_time_used` is the total time needed for the optimization procedure.
            # In case it is a surrogate, we don't want to include the time that was required to evaluate the surrogate
            total_time_used = time() - resources.start_time
            if self.is_surrogate:
                total_time_used -= total_time_used_for_objective_calls_in_s
                total_time_used += total_objective_costs

            # Time used for this configuration. The benchmark returns as cost the time of the function call +
            # the cost of the configuration. If the benchmark is a surrogate, the cost field includes the costs
            # for the function call, as well as surrogate costs. Thus, it is sufficient to use the costs returned
            # by the benchmark.
            resources.total_fuel_used += used_fuel
            resources.total_time_used_in_s = total_time_used
            resources.total_objective_costs = total_objective_costs
            resources.total_tae_calls += 1
            resources.total_time_used_for_objective_calls_in_s = total_time_used_for_objective_calls_in_s

            # Note: We update the proxy variable after checking the conditions here.
            #       This is because, we want to make sure, that this process is not be killed from outside
            #       before it was able to write the current result into the result file.
            # TODO: ADAPT THE RESOURCES WHEN THE SAME CONFIGURATION ID WAS USED.
            if not self.resource_manager.total_time_exceeds_limit(resources.total_time_used_in_s, time()) \
                    and not self.resource_manager.used_fuel_exceeds_limit(resources.total_fuel_used) \
                    and not self.resource_manager.tae_exceeds_limit(resources.total_tae_calls) \
                    and not self.resource_manager.time_per_config_exceeds_limit(result_dict['cost']):

                record = Record(start_time=start_time,
                                finish_time=finish_time,
                                function_value=result_dict['function_value'],
                                fidelity=fidelity,
                                cost=result_dict['cost'],
                                configuration=configuration,
                                configuration_id=random_config_id,
                                info=result_dict['info'],
                                function_call=resources.total_tae_calls,
                                total_time_used=resources.total_time_used_in_s,
                                total_objective_costs=resources.total_objective_costs,
                                total_fuel_used=resources.total_fuel_used)

                self.write_line_to_file(file=self.run_history if not validate else self.validated_run_history,
                                        dict_to_store=record.get_dictionary())
            else:
                logger.info('We have reached a time limit. We do not write the current record into the trajectory.')

            self.resource_manager.set_resources_without_lock(resources)

        return result_dict

    def objective_function(self,
                           configuration_id: str,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        @concurrent.process(timeout=self.resource_manager.limits.cutoff_limit_in_s)
        def __objective_function(configuration, fidelity, **benchmark_settings_for_sending):

            benchmark_settings_for_sending = self.__update_settings_for_sending(benchmark_settings_for_sending)
            benchmark = self.benchmark_partial()
            result_dict = benchmark.objective_function(configuration=configuration,
                                                       fidelity=fidelity,
                                                       **benchmark_settings_for_sending)
            return result_dict

        result_dict = self.keep_track(future_result=__objective_function,
                                      random_config_id=configuration_id,
                                      configuration=configuration,
                                      fidelity=fidelity,
                                      validate=False,
                                      rng=rng,
                                      **kwargs)

        return result_dict

    def objective_function_test(self,
                                configuration_id: str,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        @concurrent.process(timeout=self.resource_manager.limits.cutoff_limit_in_s)
        def __objective_function(configuration, fidelity, **benchmark_settings_for_sending):
            benchmark_settings_for_sending = self.__update_settings_for_sending(benchmark_settings_for_sending)
            benchmark = self.benchmark_partial()
            result_dict = benchmark.objective_function_test(configuration=configuration,
                                                            fidelity=fidelity,
                                                            **benchmark_settings_for_sending)
            return result_dict

        result_dict = self.keep_track(future_result=__objective_function,
                                      random_config_id=configuration_id,
                                      configuration=configuration,
                                      fidelity=fidelity,
                                      validate=True,
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
    def __update_settings_for_sending(benchmark_settings_for_sending):
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
        return send

    @staticmethod
    def __extract_fuel_from_fidelity(fidelity: Dict, **kwargs) -> Union[int, float]:
        if len(fidelity) == 0:
            used_fuel = 0
        elif len(fidelity) == 1:
            used_fuel = list(fidelity.values())[0]
        elif 'main_fidelity' in kwargs:
            used_fuel = fidelity[kwargs['main_fidelity']]
        else:
            # TODO: Talk about multi-multi-fidelity approaches
            raise NotImplementedError('Currently, we don"t support multi-multi-fidelity without specifying '
                                      'a main fidelity.')
        return used_fuel

    @staticmethod
    def write_line_to_file(file, dict_to_store, mode='a+'):
        write_line_to_file(file, dict_to_store, mode)

    def shutdown(self):
        self.__del__()

    def __del__(self):
        pass
