import logging
import json

from pathlib import Path
from typing import Union, Dict

from HPOBenchExperimentUtils.resource_manager.base_manager import BaseResourceManager
from HPOBenchExperimentUtils.core.data_objects import LimitObject, ResourceObject

logger = logging.getLogger('FileBasedResourceManager')


class FileBasedResourceManager(BaseResourceManager):
    def __init__(self,
                 output_dir: Path,
                 resource_file: Path,
                 limits: LimitObject
                 ):

        super(FileBasedResourceManager, self).__init__(output_dir, limits)
        self.resource_file = resource_file

    def start(self):
        super(FileBasedResourceManager, self).start()

    def stop(self):
        if self.resource_file.exists():
            self.resource_file.unlink()
        super(FileBasedResourceManager, self).stop()

    def has_started(self) -> bool:
        """If the resource file exists, then someone has already called the start() function. """
        return self.resource_file.exists()

    def get_used_resources(self) -> ResourceObject:
        resource_lock = self.get_lock()
        with resource_lock:
            resources = self.get_used_resources_without_lock()
        return resources

    def get_used_resources_without_lock(self) -> ResourceObject:
        if not self.resource_file.exists():
            default_resources = self.get_default_resources()
            self.set_resources_without_lock(default_resources)
            logger.info('Could not find resource file. Create a new one with the default resources here: '
                        f'{self.resource_file}.')
            return default_resources

        with self.resource_file.open('r') as fh:
            resources = json.load(fh)

        resources = ResourceObject(**resources)

        logger.debug(f'Retrieved resources: {resources}')
        return resources

    def set_resources(self,
                      resources: ResourceObject):
        resource_lock = self.get_lock()
        with resource_lock:
            self.set_resources_without_lock(resources)

    def set_resources_without_lock(self,
                                   resources: ResourceObject):
        with self.resource_file.open('w') as fp:
            json.dump(obj=resources.get_dictionary(), fp=fp, indent=4)

    def increase_resources(self,
                           time_used_delta: Union[int, float, None] = 0,
                           tae_calls_delta: Union[int, None] = 0,
                           fuel_used_delta: Union[int, float, None] = 0,
                           objective_costs_delta: Union[int, float, None] = 0,
                           time_used_for_objective_call_delta: Union[int, float, None] = 0) -> None:
        resource_lock = self.get_lock()
        with resource_lock:
            self.increase_resources_without_lock(time_used_delta,
                                                 tae_calls_delta,
                                                 fuel_used_delta,
                                                 objective_costs_delta)

    def increase_resources_without_lock(self,
                                        time_used_delta: Union[int, float, None] = 0,
                                        tae_calls_delta: Union[int, None] = 0,
                                        fuel_used_delta: Union[int, float, None] = 0,
                                        objective_costs_delta: Union[int, float, None] = 0,
                                        time_used_for_objective_call_delta: Union[int, float, None] = 0) -> None:

        current_resources = self.get_used_resources_without_lock()
        current_resources.add_delta(time_used_delta=time_used_delta, tae_calls_delta=tae_calls_delta,
                                    fuel_used_delta=fuel_used_delta, objective_costs_delta=objective_costs_delta,
                                    time_used_for_objective_call_delta=time_used_for_objective_call_delta)
        self.set_resources_without_lock(current_resources)

    def get_lock(self):
        return super(FileBasedResourceManager, self).get_lock()

    def total_time_exceeds_limit(self, total_time_proxy, start_time):
        return super(FileBasedResourceManager, self).total_time_exceeds_limit(total_time_proxy, start_time)

    def used_fuel_exceeds_limit(self, total_fuel_used_proxy):
        return super(FileBasedResourceManager, self).used_fuel_exceeds_limit(total_fuel_used_proxy)

    def tae_exceeds_limit(self, total_tae_calls_proxy):
        return super(FileBasedResourceManager, self).tae_exceeds_limit(total_tae_calls_proxy)

    def time_per_config_exceeds_limit(self, time_per_config):
        return super(FileBasedResourceManager, self).time_per_config_exceeds_limit(time_per_config)


__all__ = [FileBasedResourceManager]
