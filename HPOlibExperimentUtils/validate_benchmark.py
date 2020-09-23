import argparse
import json
import logging
from multiprocessing import Value
from pathlib import Path
from typing import Union, Dict

from hpolib.util.example_utils import set_env_variables_to_use_only_one_core

from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from HPOlibExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
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

    The benchmarks are by default stored in singularity container which are downloaded at the first run.

    The validation script reads in the trajectory files in the output directory. As a note, the validation script reads
    in all the trajectory files found in the given output path.
    Then all configurations are validated and written via the bookkeeper into a runhistory file.

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
    logger.info(f'Start validating procedure on benchmark {benchmark}')

    output_dir = Path(output_dir)

    assert output_dir.is_dir(), f'Result folder doesn\"t exist: {output_dir}'

    benchmark_settings = get_benchmark_settings(benchmark)
    # benchmark_settings_for_sending = prepare_dict_for_sending(benchmark_settings)

    # First, try to load already extracted trajectory file
    unvalidated_trajectories = list(output_dir.rglob(f'hpolib_trajectory.txt'))
    assert len(unvalidated_trajectories) >= 1

    if len(unvalidated_trajectories) > 1:
        logger.warning('More than one trajectory file found. Start to combine all found configurations')

    found_results = []
    for unvalidated_trajectory in unvalidated_trajectories:
        # Read in trajectory:
        with unvalidated_trajectory.open('r') as fh:
            lines = fh.readlines()

        trajectory = [json.loads(line) for line in lines]
        boot_time = trajectory[0]
        trajectory = trajectory[1:]
        found_results += trajectory

    configurations = [entry['configuration'] for entry in found_results]

    # TODO: This mapping is not needed atm. But maybe if we want to create a trajectory according to the unvalidated
    #  configurations, a mapping from config to config id (position in the trajectory) might be useful.
    # config_to_config_id = {}
    # config_id_to_config = {}
    # counter = 0
    # for config in configurations:
    #     if str(config) not in config_to_config_id:
    #         config_to_config_id[str(config)] = counter
    #         config_id_to_config[counter] = str(config)
    #         counter += 1

    configurations_to_validate = {str(configuration): -1234 for configuration in configurations}

    # Load and instantiate the benchmark
    benchmark_obj = load_benchmark(benchmark_name=benchmark_settings['import_benchmark'],
                                   import_from=benchmark_settings['import_from'],
                                   use_local=use_local)

    if use_local:
        benchmark = benchmark_obj(rng=rng, **benchmark_params)
    else:
        from hpolib import config_file
        container_source = config_file.container_source
        benchmark = benchmark_obj(rng=rng, container_source=container_source, **benchmark_params)

    total_time_proxy = Value('f', 0)

    # There is no reason why we want to have a wallclock time limit here.
    # Only a cutoff time limit seems to be useful.
    benchmark = Bookkeeper(benchmark,
                           output_dir,
                           total_time_proxy,
                           wall_clock_limit_in_s=None,
                           cutoff_limit_in_s=benchmark_settings['cutoff_in_s'],
                           is_surrogate=benchmark_settings['is_surrogate'],
                           validate=True)

    logger.debug(f'Benchmark initialized. Additional benchmark parameters {benchmark_params}')

    logger.info(f'Going to validate {len(configurations)} configuration')
    for i_config, configuration in enumerate(configurations):
        logger.debug(f'[{i_config + 1:4d}|{len(configurations):4d}] Evaluate configuration')
        config_str = str(configuration)
        if configurations_to_validate[config_str] != -1234:
            continue

        # The bookkeeper writes the trajectory automatically in to a file. But this file then contains only
        # a single entry per configuration. And the configurations are also not in the same order as the "original"
        # trajectory.
        # benchmark.objective_function_test(configuration, rng=rng)
        result_dict = benchmark.objective_function_test(configuration, rng=rng)
        configurations_to_validate[config_str] = result_dict['function_value']

    # TODO: do something with the trajectory
    # validated_trajectory = output_dir / 'hpolib_runhistory_validation.txt'
    # with validated_trajectory.open('r') as fh:
    #     lines = fh.readlines()
    #     validated_results = [json.loads(line) for line in lines]
    #     validated_results = [result for result in validated_results if 'boot_time' not in result]
    benchmark.__del__()
    return 1


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
