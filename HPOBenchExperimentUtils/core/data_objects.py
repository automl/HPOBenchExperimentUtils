from typing import Union, Dict, Any


class BaseObject(object):
    def get_dictionary(self):
        raise NotImplementedError()

    def __repr__(self):
        return f' \n '.join((f'{key}: {value}' for key, value in self.get_dictionary().items()))


class Record(BaseObject):
    def __init__(self,
                 start_time: Union[int, float, None] = None,
                 finish_time: Union[int, float, None] = None,
                 function_value: Union[int, float, None] = None,
                 fidelity: Union[Dict, None] = None,
                 cost: Union[int, float, None] = None,
                 configuration: Union[Dict, None] = None,
                 info: Union[Dict, None] = None,
                 function_call: Union[int, None] = None,
                 total_time_used: Union[int, float, None] = None,
                 total_objective_costs: Union[int, float, None] = None,
                 total_fuel_used: Union[int, float, None] = None,
                 configuration_id: Union[str, None, None] = None):

        self.start_time = start_time
        self.finish_time = finish_time
        self.function_value = function_value
        self.fidelity = fidelity
        self.cost = cost
        self.configuration = configuration
        self.info = info
        self.function_call = function_call
        self.total_time_used = total_time_used
        self.total_objective_costs = total_objective_costs
        self.total_fuel_used = total_fuel_used
        self.configuration_id = configuration_id

    def get_dictionary(self):
        return {'start_time': self.start_time,
                'finish_time': self.finish_time,
                'function_value': self.function_value,
                'fidelity': self.fidelity,
                'cost': self.cost,
                'configuration': self.configuration,
                'configuration_id': self.configuration_id,
                'info': self.info,
                'function_call': self.function_call,
                'total_time_used': self.total_time_used,
                'total_objective_costs': self.total_objective_costs,
                'total_fuel_used': self.total_fuel_used,
                }


class LimitObject(BaseObject):
    def __init__(self,
                 time_limit_in_s: Union[int, None] = None,
                 tae_limit: Union[int, None] = None,
                 fuel_limit: Union[int, float, None] = None,
                 cutoff_limit_in_s: Union[int, float, None] = None,
                 # mem_limit_in_mb: Union[int, float, None] = None
                 ):

        self.time_limit_in_s = time_limit_in_s
        self.tae_limit = tae_limit
        self.fuel_limit = fuel_limit
        self.cutoff_limit_in_s = cutoff_limit_in_s
        # self.mem_limit_in_mb = mem_limit_in_mb

    def get_dictionary(self):
        return {'time_limit_in_s': self.time_limit_in_s,
                'tae_limit': self.tae_limit,
                'fuel_limit': self.fuel_limit,
                'cutoff_limit_in_s': self.cutoff_limit_in_s,
                # 'mem_limit_in_mb': self.mem_limit_in_mb
                }


class ResourceObject(BaseObject):
    keys = ['total_time_used_in_s',
            'total_tae_calls',
            'total_fuel_used',
            'total_objective_costs',
            'total_time_used_for_objective_calls_in_s',
            'start_time']

    def __init__(self,
                 total_time_used_in_s,
                 total_tae_calls,
                 total_fuel_used,
                 total_objective_costs,
                 total_time_used_for_objective_calls_in_s,
                 start_time
                 ):
        self.total_time_used_in_s = total_time_used_in_s
        self.total_tae_calls = total_tae_calls
        self.total_fuel_used = total_fuel_used
        self.total_objective_costs = total_objective_costs
        self.total_time_used_for_objective_calls_in_s = total_time_used_for_objective_calls_in_s
        self.start_time = start_time

    def get_dictionary(self):
        return {'total_time_used_in_s': self.total_time_used_in_s,
                'total_tae_calls': self.total_tae_calls,
                'total_fuel_used': self.total_fuel_used,
                'total_objective_costs': self.total_objective_costs,
                'total_time_used_for_objective_calls_in_s': self.total_time_used_for_objective_calls_in_s,
                'start_time': self.start_time}

    def add_delta(self,
                  time_used_delta: Union[int, float, None] = None,
                  tae_calls_delta: Union[int, None] = None,
                  fuel_used_delta: Union[int, float, None] = None,
                  objective_costs_delta: Union[int, float, None] = None,
                  time_used_for_objective_call_delta: Union[int, float, None] = None) -> None:

        if time_used_delta is not None:
            if self.total_time_used_in_s is None:
                self.total_time_used_in_s = 0
            self.total_time_used_in_s += time_used_delta

        if tae_calls_delta is not None:
            if self.total_tae_calls is None:
                self.total_tae_calls = 0
            self.total_tae_calls += tae_calls_delta

        if fuel_used_delta is not None:
            if self.total_fuel_used is None:
                self.total_fuel_used = 0
            self.total_fuel_used += fuel_used_delta

        if objective_costs_delta is not None:
            if self.total_objective_costs is None:
                self.total_objective_costs = 0
            self.total_objective_costs += objective_costs_delta

        if time_used_for_objective_call_delta is not None:
            if self.total_time_used_for_objective_calls_in_s is None:
                self.total_time_used_for_objective_calls_in_s = 0
            self.total_time_used_for_objective_calls_in_s += objective_costs_delta
