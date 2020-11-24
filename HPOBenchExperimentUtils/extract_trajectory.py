import json
import logging
import os

from pathlib import Path
from typing import Union, List, Dict

from HPOBenchExperimentUtils.core.trajectories import create_trajectory
from HPOBenchExperimentUtils.utils.validation_utils import load_json_files

from HPOBenchExperimentUtils.utils import TRAJECTORY_V1_FILENAME, TRAJECTORY_V2_FILENAME, RUNHISTORY_FILENAME

from HPOBenchExperimentUtils import _log as _main_log, _default_log_format

_main_log.setLevel(level=logging.INFO)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


def extract_trajectory(output_dir: Union[Path, str], debug: Union[bool, None] = False):
    """
    TODO:

    STEPS:
    ------
    1) Load all runhistories from the optimization step.
        -> Search for the runhistory files recursively. Perform all other steps for all runhistories.
    2) Iterate through the histories.
        3) Read in all runs
        4) Extract Trajectory 1: Lower is better and larger budget is better.
        5) Extract Trajectory 2: Ignore Budget, only lower is better.
        6) Save both trajectories to file
    7) THE END.


    Parameters
    ----------
    output_dir : str, Path
        Directory where the optimizer stored its results.

    debug: bool, None
        Enables the debug message logging.
    """

    _log.info('Start extracting the trajectories')

    if debug:
        _main_log.setLevel(level=logging.DEBUG)

    output_dir = Path(output_dir)
    assert output_dir.is_dir(), f'Result folder doesn\'t exist: {output_dir}'

    # Search all runhistories in the output directory
    runhistory_paths = list(output_dir.rglob(RUNHISTORY_FILENAME))

    # Load the runhistories
    runhistories = load_json_files(runhistory_paths)

    def print_traj(trajectory):
        for i, r in enumerate(trajectory):
            if i != 0:
                print(f'{i:2d}, {r["function_call"]:3d}, {r["function_value"]:.15f},\t {list(r["fidelity"].values())[0]}')

    for i_rh, (runhistory, runhistory_path) in enumerate(zip(runhistories, runhistory_paths)):

        trajectory = create_trajectory(runhistory, bigger_is_better=True)
        # print_traj(trajectory)
        write_list_of_dicts_to_file(runhistory_path.parent / TRAJECTORY_V1_FILENAME, trajectory)

        trajectory = create_trajectory(runhistory, bigger_is_better=False)
        # print_traj(trajectory)
        write_list_of_dicts_to_file(runhistory_path.parent / TRAJECTORY_V2_FILENAME, trajectory)

    return 1


def write_list_of_dicts_to_file(output_file: Path, data: List[Dict]):
    with output_file.open('w') as fh:
        for dict_to_store in data:
            json.dump(dict_to_store, fh)
            fh.write(os.linesep)


if __name__ == "__main__":

    extract_trajectory(output_dir='/home/philipp/Dokumente/Code/HPOlibExperimentUtils/experiments/cartpole/',
                       debug=True)

    # parser = argparse.ArgumentParser(prog='HPOBench Wrapper',
    #                                  description='HPOBench validated a trajectory from a benchmark with a '
    #                                              'unified interface',
    #                                  )
    #
    # parser.add_argument('--output_dir', required=True, type=str)
    # parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    # parser.add_argument('--rng', required=False, default=0, type=int)
    # parser.add_argument('--recompute_all', action='store_true', default=False)
    # parser.add_argument('--use_local', action='store_true', default=False)
    # parser.add_argument('--debug', action='store_true', default=False, help="When given, enables debug mode logging.")
    #
    # args, unknown = parser.parse_known_args()
    # benchmark_params = transform_unknown_params_to_dict(unknown)
    #
    # validate_benchmark(**vars(args), **benchmark_params)


