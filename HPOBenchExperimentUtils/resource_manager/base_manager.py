import logging
import shutil
from pathlib import Path
from time import time
from typing import Union, Dict

from oslo_concurrency import lockutils

logger = logging.getLogger('BaseResourceManager')
from HPOBenchExperimentUtils.core.data_objects import LimitObject, ResourceObject


class BaseResourceManager:

    def __init__(self,
                 output_dir: Path,
                 limits: LimitObject
                 ):

        self.lock_dir = output_dir / 'lock_dir'
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        self.lock_name = 'resource_lock'
        self.limits = limits

    def start(self):
        _ = self.get_used_resources()
        logger.info('ResourceManager started')

    def stop(self):
        if (self.lock_dir / self.lock_name).exists():
            (self.lock_dir / self.lock_name).unlink()

        # Check if the lock dir is empty
        if self.lock_dir.exists() and not any(self.lock_dir.iterdir()):
            shutil.rmtree(self.lock_dir)

    def has_started(self) -> bool:
        raise NotImplementedError()

    def get_used_resources(self) -> ResourceObject:
        raise NotImplementedError()

    def get_used_resources_without_lock(self) -> ResourceObject:
        raise NotImplementedError()

    def set_resources(self,
                      resources: ResourceObject):
        raise NotImplementedError()

    def set_resources_without_lock(self,
                                   resources: ResourceObject):
        raise NotImplementedError()

    def increase_resources(self,
                           time_used_delta: Union[int, float, None] = 0,
                           tae_calls_delta: Union[int, None] = 0,
                           fuel_used_delta: Union[int, float, None] = 0,
                           objective_costs_delta: Union[int, float, None] = 0,
                           time_used_for_objective_call_delta: Union[int, float, None] = 0) -> None:
        raise NotImplementedError()

    def increase_resources_without_lock(self,
                                        time_used_delta: Union[int, float, None] = 0,
                                        tae_calls_delta: Union[int, None] = 0,
                                        fuel_used_delta: Union[int, float, None] = 0,
                                        objective_costs_delta: Union[int, float, None] = 0,
                                        time_used_for_objective_call_delta: Union[int, float, None] = 0) -> None:
        raise NotImplementedError()

    def get_lock(self):
        return lockutils.lock(name=self.lock_name, external=True, do_log=False,
                              lock_path=str(self.lock_dir), delay=0.01)

    def total_time_exceeds_limit(self, total_time_proxy, start_time) -> bool:
        return (self.limits.time_limit_in_s is not None
                and (total_time_proxy > self.limits.time_limit_in_s
                     or time() - start_time > self.limits.time_limit_in_s))

    def used_fuel_exceeds_limit(self, total_fuel_used_proxy) -> bool:
        return self.limits.fuel_limit is not None and total_fuel_used_proxy > self.limits.fuel_limit

    def tae_exceeds_limit(self, total_tae_calls_proxy) -> bool:
        return self.limits.tae_limit is not None and total_tae_calls_proxy > self.limits.tae_limit

    def time_per_config_exceeds_limit(self, time_per_config) -> bool:
        return time_per_config > self.limits.cutoff_limit_in_s

    @staticmethod
    def get_default_resources() -> ResourceObject:
        default_resources = ResourceObject(total_time_used_in_s=0.0,
                                           total_tae_calls=0,
                                           total_fuel_used=0.0,
                                           total_objective_costs=0.0,
                                           total_time_used_for_objective_calls_in_s=0.0,
                                           start_time=time())
        return default_resources


__all__ = [BaseResourceManager]
