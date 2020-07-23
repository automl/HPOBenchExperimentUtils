import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict

from hpolib.util.example_utils import set_env_variables_to_use_only_one_core

from HPOlibExperimentUtils import BOHBReader, SMACReader
from HPOlibExperimentUtils.utils.optimizer_utils import parse_fidelity_type, prepare_dict_for_sending
from HPOlibExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_setting_per_benchmark, \
    load_benchmark, get_benchmark_names

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('BenchmarkValidation')

set_env_variables_to_use_only_one_core()


def validate_benchmark(benchmark: str,
                       output_dir: Union[Path, str],
                       rng: int,
                       use_local: Union[bool, None] = False,
                       **benchmark_params: Dict):
    """
    After running a optimization run, often we want to validate the found configurations. To be more precisely, we want
    to validate (run on the highest budget) the configurations, which were incumbents for some time during the
    optimization process.

    Similar to run_benchmark(), only SMAC, BOHB and Dragonfly are available as Optimizer.
    The benchmarks are by default stored in singularity container which are downloaded at the first run.

    The validation script automatically detects the used optimizer and reads in the trajectory files in the output
    directory.

    Note:
    -----
    If there are more than one trajectory files found in the `output_dir`, the first one is selected for validation.
    This will change in the future. Then all found trajectories are combined and validated.

    Parameters
    ----------
    benchmark : str
        This benchmark is selected from the HPOlib3 and executed. with the optimizer from above.
        Please have a look at the experiment_settings.json file to see what benchmarks are available.
        Incomplete list of supported benchmarks:
        'cartpolereduced', 'cartpolefull',
        'xgboost',
        'learna', 'metalearna',
        'NASCifar10ABenchmark', 'NASCifar10BBenchmark', 'NASCifar10CBenchmark',
        'SliceLocalizationBenchmark', 'ProteinStructureBenchmark',
            'NavalPropulsionBenchmark', 'ParkinsonsTelemonitoringBenchmark'
    output_dir : str, Path
        Directory where the optimizer stored its results. Here should also be the trajectory file,
        which was generated in the previous step.
    rng : int, None
        Random seed for the experiment. Also changes the output directory. By default 0.
    use_local : bool, None
        If you want to use the HPOlib3 benchamrks in a non-containerizd version (installed locally inside the
        current python environment), you can set this parameter to True. This is not recommend.
    benchmark_params : Dict
        Some benchmarks take special parameters for the initialization. For example, The XGBOostBenchmark takes as
        input a task_id. This task_id specifies the OpenML dataset to use.

        Please take a look into the HPOlib3 Benchamarks to find out if the benchmark needs further parameter.
        Note: Most of them dont need further parameter.
    """

    output_dir = Path(output_dir)

    assert output_dir.is_dir(), f'Result folder doesn\"t exist: {output_dir}'
    optimizer_settings, benchmark_settings = get_setting_per_benchmark(benchmark, rng=rng, output_dir=output_dir)
    benchmark_settings_for_sending = prepare_dict_for_sending(benchmark_settings)

    # First, try to load already extracted trajectory file
    unvalidated_trajectory = list(output_dir.rglob(f'traj_hpolib.json'))
    if len(unvalidated_trajectory) == 0:
        unvalidated_trajectory = output_dir
        already_extracted = False
    elif len(unvalidated_trajectory) == 1:
        unvalidated_trajectory = unvalidated_trajectory[0]
        already_extracted = True
    else:
        # TODO: Support selecting multiple trajecotry files.
        logger.warning('Found mulitple trajectory files in the directory. We select the first.')
        unvalidated_trajectory = unvalidated_trajectory[0]
        already_extracted = True

    # Check if it was a BOHB run. In this case we find a results.json file in the output directory.
    bohb_run = len(list(unvalidated_trajectory.parent.glob('**/results.json'))) != 0

    logger.debug(f'Unvalidated Trajectory: {unvalidated_trajectory}\n'
                 f'Already extracted {already_extracted} - Bohb run: {bohb_run}')

    # TODO: Adapt to Dragonfly usage
    # Instantiate the reader. This reader can parse the trajectory file from the optimization run.
    reader = SMACReader() if already_extracted or not bohb_run else BOHBReader()
    reader.read(file_path=unvalidated_trajectory)
    reader.get_trajectory()
    logger.debug(reader)

    # Read in the configurations which where found in the trajectory file.
    # In a later step, we want to validate those configurations. This means to run them again on the highest budget.
    trajectory_ids = reader.get_configuration_ids_trajectory()
    configurations_to_validate = \
        OrderedDict({traj_id: reader.config_ids_to_configs[traj_id] for traj_id in trajectory_ids})

    validated_loss = OrderedDict({traj_id: -1234 for traj_id in trajectory_ids})

    # Load and instantiate the benchmark
    benchmark_obj = load_benchmark(benchmark_settings, use_local)
    benchmark = benchmark_obj(container_source='library://phmueller/automl',
                              **benchmark_params)

    # Now, we start the validation process for every configuration.
    # Note: This process may take a lot of time.
    for i, traj_id in enumerate(configurations_to_validate):
        config = configurations_to_validate[traj_id]
        max_budget = optimizer_settings['max_budget']
        cast_to = parse_fidelity_type(benchmark_settings['fidelity_type'])
        fidelity = {benchmark_settings['fidelity_name']: cast_to(max_budget)}
        logger.debug(f'Validate configuration: {config}\n'
                     f'Max Budget: {max_budget} on fidelity: {benchmark_settings["fidelity_name"]}')

        result_dict = benchmark.objective_function_test(config, **fidelity, **benchmark_settings_for_sending)
        validated_loss[traj_id] = result_dict['function_value']
        logger.debug(f'Validated config [{i+1:5d}|{len(configurations_to_validate)}]: {validated_loss[traj_id]}')

    # We add the validated trajectory to the reader. The reader then can export it in the SMAC-like trajectory format.
    reader.add_validated_trajectory(validated_loss)
    traj_path = optimizer_settings['output_dir'] / f'traj_validated_hpolib.json'

    reader.export_validated_trajectory(traj_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 validated a trajectory from a benchmark with a '
                                                 'unified interface',
                                     )

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)

    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    validate_benchmark(**vars(args), **benchmark_params)
