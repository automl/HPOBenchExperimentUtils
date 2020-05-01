from typing import List, Union, Dict, Any


class Run(object):
    def __init__(self):
        self.config_id = None
        self.relative_finish_time = 0
        self.loss = 0
        self.budget = 0
        self.info = {}

    def get_relative_finish_time(self):
        return self.relative_finish_time

    def set_values_bohb(self, config_id: List, budget: Union[float, int], time_stamps: Dict, result: Dict,
                        exception: Any, global_start_time: Union[float, int], info: Dict):
        self.config_id = tuple(config_id)
        self.budget = budget
        finish_time = time_stamps.get('finished')
        self.relative_finish_time = finish_time - global_start_time
        self.loss = result.get('loss')
        self.info = info

    def set_values_smac(self, config_id: int, budget: Union[int, float], wallclock_time: Union[int, float],
                        cost: Union[int, float], info: Dict):
        self.config_id = config_id
        self.budget = budget
        self.relative_finish_time = wallclock_time
        self.loss = cost
        self.info = info

    def __repr__(self):
        return f'Run(ID: {self.config_id}, Budget: {self.budget}, Loss: {self.loss}, ' \
               f'Wall-clock time {self.relative_finish_time}, Info: {self.info}'
