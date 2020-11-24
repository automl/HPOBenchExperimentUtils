import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from HPOBenchExperimentUtils.utils import VALIDATED_TRAJECTORY_V1_FILENAME, VALIDATED_TRAJECTORY_V2_FILENAME, TRAJECTORY_V1_FILENAME

_log = logging.getLogger(__name__)


def write_validated_trajectory(unvalidated_traj: List, validation_results: Dict, unvalidated_traj_path: Path):
    """
    Make a copy of the unvalidated trajectory. Then replace the function values in the copy with their validated
    function value.
    Store the validated trajectory in the same directory.

    Parameters
    ----------
    unvalidated_traj
    validation_results
    unvalidated_traj_path
    """

    # Update the function values
    validated_trajectory = []
    for i_entry, entry in enumerate(unvalidated_traj):
        # Skip the first entry. That's the boot time entry.
        if i_entry != 0:
            config = str(entry['configuration'])

            result_dict = validation_results[config]

            entry['function_value'] = result_dict['function_value']

            # TODO: PM: Discuss which fields need to be overwritten.
            # entry['info']['fidelity'] = result_dict['info']['fidelity']
            # entry['fidelity'] = result_dict['fidelity']

        validated_trajectory.append(entry)

    # Write back the validated trajectory
    if unvalidated_traj_path.name == TRAJECTORY_V1_FILENAME:
        name = VALIDATED_TRAJECTORY_V1_FILENAME
    else:
        name = VALIDATED_TRAJECTORY_V2_FILENAME

    validated_trajectory_path = unvalidated_traj_path.parent / name
    with validated_trajectory_path.open('w') as fh:
        for dict_to_store in validated_trajectory:
            json.dump(dict_to_store, fh)
            fh.write(os.linesep)


def extract_configs_from_trajectories(trajectories: List) -> List:
    """
    This function collects all configurations in a given list of trajectories. The structure will not be changed. This
    means that the function returns the configurations per trajectory.

    Parameters
    ----------
    trajectories : List of trajectories.
        Note: Trajectories are also represented as list.

    Returns
    -------
        List
    """
    """ Return a List of all unvalidated configurations from the collecetd trajectory files. """
    configurations = [run['configuration'] for trajectory in trajectories for run in trajectory[1:]]
    return configurations


def load_json_files(file_paths: List[Path]) -> List:
    """
    Read in all json formatted files. The input could be a list of paths of runhistories.

    Parameters
    ----------
    file_paths: List[Path]

    Returns
    -------
    List
        List of lists. Each list contains the content of a json file.
    """
    assert len(file_paths) >= 1

    data = []
    for file in file_paths:
        lines = read_lines(file)

        file_content = [json.loads(line) for line in lines]
        data.append(file_content)
    return data


def load_configs_with_function_values_from_runhistories(file_paths: List[Path]):

    data = load_json_files(file_paths)

    configs_dict = {}

    for runhistory in data:
        for entry in runhistory:
            if 'boot_time' in entry:
                continue

            config = str(entry['configuration'])
            func_value = entry['function_value']
            if config not in configs_dict or configs_dict[config] > func_value:
                configs_dict[config] = entry

    return configs_dict


def load_validated_configurations(output_dir: Path) -> Dict:
    # Try to find previously computed validation run histories. Look also in subfolder for them.
    validated_runhistory_paths = list(output_dir.rglob(VALIDATED_RUNHISTORY_FILENAME))
    validated_configs = {}
    for rh_path in validated_runhistory_paths:
        lines = read_lines(rh_path)

        for line in lines:
            entry = json.loads(line)
            if 'boot_time' in entry:
                continue

            config = str(entry['configuration'])
            func_value = entry['function_value']
            if config not in validated_configs or validated_configs[config] > func_value:
                validated_configs[config] = func_value

    return validated_configs


def read_lines(file: Path) -> List:
    """ Read multiple lines from a file"""
    with file.open('r') as fh:
        lines = fh.readlines()
    return lines


def load_trajectories_as_df(input_dir, which="test"):
    if which == "train":
        trajectories_paths = list(input_dir.rglob(f'hpobench_trajectory.txt'))
    elif which == "test":
        trajectories_paths = list(input_dir.rglob(f'hpobench_trajectory_validated.txt'))
    elif which == "runhistory":
        trajectories_paths = list(input_dir.rglob(f'hpobench_runhistory.txt'))
    else:
        raise ValueError(f'Specified parameter must be one of [train, test, runistory] but was {which}')
    unique_optimizer = defaultdict(lambda: [])
    for path in trajectories_paths:
        opt = path.parent.parent.name
        unique_optimizer[opt].append(path)
    return unique_optimizer


def get_statistics_df(optimizer_df):
    select_cols = ['mean', 'std', 'median', 'q25', 'q75', 'mean_inf', 'up', 'lo']
    # Dataframe for the learning curves per optimizer
    piv = optimizer_df.pivot(index='total_time_used', columns='id', values='function_values')

    piv = piv.fillna(method='ffill')
    piv = piv.fillna(method='bfill')

    piv['mean'] = piv.mean(axis=1)
    piv['std'] = piv.std(axis=1)

    piv['median'] = piv.median(axis=1)
    piv['q25'] = piv.quantile(0.25, axis=1)
    piv['q75'] = piv.quantile(0.75, axis=1)

    piv['lo'] = piv.min(axis=1)
    piv['up'] = piv.max(axis=1)

    piv['mean_inf'] = np.minimum.accumulate(piv['mean'])
    piv['lo'] = np.minimum.accumulate(piv['lo'])
    piv['up'] = np.minimum.accumulate(piv['up'])

    piv = piv[select_cols]
    return piv


def df_per_optimizer(key, unvalidated_trajectories, y_best: float=0):
    optimizer_df = pd.DataFrame()

    if y_best != 0:
        _log.info("Found y_best = %g; Going to compute regret" % y_best)

    for id, traj in enumerate(unvalidated_trajectories):
        trajectory_df = pd.DataFrame(columns=['optimizer', 'id',
                                              'function_values', 'fidelity_value',
                                              'total_time_used', 'total_objective_costs'])
        function_values = [record['function_value']-y_best for record in traj[1:]]
        total_time_used = [record['total_time_used'] for record in traj[1:]]
        total_obj_costs = [record['total_objective_costs'] for record in traj[1:]]
        costs = [record['cost'] for record in traj[1:]]
        start = [record['start_time'] for record in traj[1:]]
        finish = [record['finish_time'] for record in traj[1:]]

        # this is a dict with only one entry
        fidel_values = [record['fidelity'][list(record['fidelity'])[0]] for record in traj[1:]]

        trajectory_df['optimizer'] = [key for _ in range(len(traj[1:]))]
        trajectory_df['id'] = [id for _ in range(len(traj[1:]))]
        trajectory_df['total_time_used'] = total_time_used
        trajectory_df['total_objective_costs'] = total_obj_costs
        trajectory_df['function_values'] = function_values
        trajectory_df['fidel_values'] = fidel_values
        trajectory_df['costs'] = costs
        trajectory_df['start_time'] = start
        trajectory_df['finish_time'] = finish

        optimizer_df = pd.concat([optimizer_df, trajectory_df])
    return optimizer_df
