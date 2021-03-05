from typing import Union, Dict, Any


class Record(object):
    def __init__(self, start_time = None, finish_time = None, function_value = None,
                 fidelity = None, cost = None, configuration = None,
                 info: Dict = None, function_call: int = None,
                 total_time_used: Union[int, float] = None, total_objective_costs: Union[int, float] = None,
                 total_fuel_used: Union[int, float] = None):

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

    def get_dictionary(self):
        return {'start_time': self.start_time,
                'finish_time': self.finish_time,
                'function_value': self.function_value,
                'fidelity': self.fidelity,
                'cost': self.cost,
                'configuration': self.configuration,
                'info': self.info,
                'function_call': self.function_call,
                'total_time_used': self.total_time_used,
                'total_objective_costs': self.total_objective_costs,
                'total_fuel_used': self.total_fuel_used,
                }
