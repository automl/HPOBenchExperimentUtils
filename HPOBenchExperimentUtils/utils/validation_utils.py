import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def write_validated_trajectory(unvalidated_traj: List, validation_results: Dict, unvalidated_traj_path: Path):
    """ update the function value in the unvalidated trajectory with the validated function value.
        Then write the validated trajectory in a file, which is in the same folder as the unvalidated trajectory
        file """

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
    validated_trajectory_path = unvalidated_traj_path.parent / 'hpobench_trajectory_validated.txt'
    with validated_trajectory_path.open('w') as fh:
        for dict_to_store in validated_trajectory:
            json.dump(dict_to_store, fh)
            fh.write(os.linesep)


def get_unvalidated_configurations(trajectories: List) -> List:
    """ Return a List of all unvalidated configurations from the collecetd trajectory files. """
    configurations = [run['configuration'] for trajectory in trajectories for run in trajectory[1:]]
    return configurations


def load_trajectories(trajectory_paths: List) -> List:
    """
    Collect the trajectories which are in the output directory

    Parameters
    ----------
    output_dir : Path
        Direcotry, where trajectory files with configurations to validate are stored. Can also be a parent
        directory. Then, read all hpobench_trajectory.txt files.

    Returns
    -------
    List
    """
    assert len(trajectory_paths) >= 1
    if len(trajectory_paths) > 1:
        _log.warning('More than one trajectory file found. Start to combine all found configurations')

    _log.info("Loading %d trajectories" % len(trajectory_paths))
    # Load all trajectories
    found_trajectories = []
    for trajectory_path in trajectory_paths:
        # Read in trajectory:
        _log.info("Reading %s" % trajectory_path)
        with trajectory_path.open("r") as fh:
            trajectory = [json.loads(line) for line in fh]
        found_trajectories.append(trajectory)
    return found_trajectories


def load_validated_configurations(output_dir: Path) -> Dict:
    # Try to find previously computed validation run histories. Look also in subfolder for them.
    validated_runhistory_paths = list(output_dir.rglob(f'hpobench_runhistory_validation.txt'))
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
    if y_best != 0:
        _log.info("Found y_best = %g; Going to compute regret" % y_best)
    _log.info("Creating DataFrame for %d inputs" % len(unvalidated_trajectories))
    dataframe = {
        "optimizer": [],
        "id": [],
        "total_time_used": [],
        "total_objective_costs": [],
        "function_values": [],
        "fidel_values": [],
        "costs": [],
        "start_time": [],
        "finish_time": [],
    }

    for id, traj in enumerate(unvalidated_trajectories):
        _log.info("Handling input with %d records for %s" % (len(traj), key))
        function_values = [record['function_value']-y_best for record in traj[1:]]
        total_time_used = [record['total_time_used'] for record in traj[1:]]
        total_obj_costs = [record['total_objective_costs'] for record in traj[1:]]
        costs = [record['cost'] for record in traj[1:]]
        start = [record['start_time'] for record in traj[1:]]
        finish = [record['finish_time'] for record in traj[1:]]

        # this is a dict with only one entry
        fidel_values = [record['fidelity'][list(record['fidelity'])[0]] for record in traj[1:]]

        dataframe["optimizer"].extend([key for _ in range(len(traj[1:]))])
        dataframe["id"].extend([id for _ in range(len(traj[1:]))])
        dataframe['total_time_used'].extend(total_time_used)
        dataframe['total_objective_costs'].extend(total_obj_costs)
        dataframe['function_values'].extend(function_values)
        dataframe['fidel_values'].extend(fidel_values)
        dataframe['costs'].extend(costs)
        dataframe['start_time'].extend(start)
        dataframe['finish_time'].extend(finish)

    dataframe = pd.DataFrame(dataframe)
    return dataframe