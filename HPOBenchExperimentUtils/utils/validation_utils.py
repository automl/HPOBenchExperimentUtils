try:
    import ujson as json
    import json as json_backup
    print("Use ujson")
except:
    import json
    print("Using json. Installing ujson could provide speedup")

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import pandas as pd

from HPOBenchExperimentUtils.utils import VALIDATED_TRAJECTORY_V1_FILENAME, VALIDATED_TRAJECTORY_V2_FILENAME, \
    VALIDATED_TRAJECTORY_V3_FILENAME, TRAJECTORY_V1_FILENAME, TRAJECTORY_V2_FILENAME, TRAJECTORY_V3_FILENAME, \
    RUNHISTORY_FILENAME, VALIDATED_RUNHISTORY_FILENAME

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

            # Update some fields with the validated (new) values.
            entry['function_value_unvalidated'] = entry['function_value']
            entry['function_value'] = result_dict['function_value']

            entry['cost_unvalidated'] = entry['cost']
            entry['cost'] = result_dict['cost']

            entry['info']['fidelity_unvalidated'] = entry['info'].get('fidelity', None)
            entry['info']['fidelity'] = result_dict['info']['fidelity']

            entry['fidelity_unvalidated'] = entry['fidelity']
            entry['fidelity'] = result_dict['fidelity']

            entry['start_time_unvalidated'] = entry['start_time']
            entry['start_time'] = result_dict['start_time']

            entry['finish_time_unvalidated'] = entry['finish_time']
            entry['finish_time'] = result_dict['finish_time']

        validated_trajectory.append(entry)

    # Write back the validated trajectory
    if unvalidated_traj_path.name == TRAJECTORY_V1_FILENAME:
        name = VALIDATED_TRAJECTORY_V1_FILENAME
    elif unvalidated_traj_path.name == TRAJECTORY_V2_FILENAME:
        name = VALIDATED_TRAJECTORY_V2_FILENAME
    elif unvalidated_traj_path.name == TRAJECTORY_V3_FILENAME:
        name = VALIDATED_TRAJECTORY_V3_FILENAME
    else:
        _log.critical(f"Unknown trajectory filename {unvalidated_traj.name}")
        raise ValueError()

    validated_trajectory_path = unvalidated_traj_path.parent / name
    with validated_trajectory_path.open('w') as fh:
        for dict_to_store in validated_trajectory:
            json.dump(dict_to_store, fh)
            fh.write(os.linesep)

    _log.info(f'Writing the trajectory to {validated_trajectory_path} was successful.')


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
    """ Return a List of all unvalidated configurations from the collected trajectory files. """
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

    start = time.time()
    assert len(file_paths) >= 1

    data = []
    for file in file_paths:
        lines = read_lines(file)
        file_content = []
        for line in lines:
            try:
                r = json.loads(line)
            except:
                try:
                    r = json_backup.loads(line)
                except:
                    raise

            file_content.append(r)
        #file_content = [json.loads(line) for line in lines]
        data.append(file_content)
    dur = time.time() - start
    _log.info("Reading %d files took %f sec" % (len(file_paths), dur))
    return data


def load_configs_with_function_values_from_runhistories(file_paths: List[Path]):

    if len(file_paths) == 0:
        return {}

    data = load_json_files(file_paths)

    configs_dict = {}
    for runhistory in data:
        for entry in runhistory:
            if 'boot_time' in entry:
                continue

            config = str(entry['configuration'])
            func_value = entry['function_value']
            if config not in configs_dict or configs_dict[config]['function_value'] > func_value:
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


def load_trajectories_as_df(input_dir, which="test_v1"):
    if which == "train_v1":
        trajectories_paths = list(input_dir.rglob(TRAJECTORY_V1_FILENAME))
    elif which == "train_v2":
        trajectories_paths = list(input_dir.rglob(TRAJECTORY_V2_FILENAME))
    elif which == "test_v1":
        trajectories_paths = list(input_dir.rglob(VALIDATED_TRAJECTORY_V1_FILENAME))
    elif which == "test_v2":
        trajectories_paths = list(input_dir.rglob(VALIDATED_TRAJECTORY_V2_FILENAME))
    elif which == "runhistory":
        trajectories_paths = list(input_dir.rglob(RUNHISTORY_FILENAME))
    else:
        raise ValueError('Specified parameter must be one of [train_v1, train_v2, test_v1, test_v2, runistory]'
                         f'but was {which}')

    unique_optimizer = defaultdict(lambda: [])
    for path in trajectories_paths:
        opt = path.parent.parent.name

        if "test" in which:
            if (path.parent / VALIDATED_TRAJECTORY_V3_FILENAME).is_file():
                _log.critical(f"Change to {path.parent / VALIDATED_TRAJECTORY_V3_FILENAME}")
                path = path.parent / VALIDATED_TRAJECTORY_V3_FILENAME

        unique_optimizer[opt].append(path)
    return unique_optimizer


def get_statistics_df(optimizer_df):
    select_cols = ['mean', 'std', 'median', 'q25', 'q75', 'mean_inf', 'up', 'lo']
    # Dataframe for the learning curves per optimizer
    piv = optimizer_df.pivot(index='total_time_used', columns='id', values='function_values')

    piv = piv.fillna(method='ffill')
    vali = -1
    for c in piv.columns:
        vali = max(vali, piv[c].first_valid_index())
    piv = piv.loc[vali:]

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
