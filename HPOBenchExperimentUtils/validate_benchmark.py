import argparse
import logging
import threading
from pathlib import Path
from typing import Union, Dict

from HPOBenchExperimentUtils import _log as _root_log
from HPOBenchExperimentUtils.core.nameserver import start_nameserver
from HPOBenchExperimentUtils.core.scheduler import Scheduler, Content
from HPOBenchExperimentUtils.core.worker import Worker
from HPOBenchExperimentUtils.utils import TRAJECTORY_V1_FILENAME, TRAJECTORY_V2_FILENAME, \
    TRAJECTORY_V3_FILENAME, VALIDATED_RUNHISTORY_FILENAME
from HPOBenchExperimentUtils.utils.pyro_utils import nic_name_to_host
from HPOBenchExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
    get_benchmark_names
from HPOBenchExperimentUtils.utils.validation_utils import write_validated_trajectory, \
    extract_configs_from_trajectories, load_json_files, load_configs_with_function_values_from_runhistories

_root_log.setLevel(level=logging.INFO)
main_logger = logging.getLogger('validate_benchmark')


def validate_benchmark(benchmark: str,
                       procedure: str,
                       output_dir: Union[Path, str],
                       run_id: int,
                       worker_id: int,
                       rng: int,
                       use_local: Union[bool, None] = False,
                       debug: Union[bool, None] = False,
                       interface: str = 'lo',
                       nameserver_port: int = 0,
                       recompute_all: bool = False,
                       **benchmark_params: Dict,
                       ):
    """
    After running an optimization run, often, we want to validate the found configurations. To be more precisely, we
    want to validate (run on the highest budget and a test set) the configurations, which were incumbents for some time
    during the optimization process.

    The benchmarks are by default stored in singularity container, which are downloaded at the first run.

    The validation script reads the trajectory files in the output directory. Note that the validation script reads
    all the trajectory files found recursively in the given output path.
    Then, all configurations are validated and written into a runhistory file.

    We write intermediate results to file, so that if a run crashed, we don't have to start from scratch. We read in
    the previously validated runhistories also recursively starting from the defined output directory.

    Parameters
    ----------
    benchmark : str
        This benchmark is selected from the HPOBench and executed. with the optimizer from above.
        Please have a look at the experiment_settings.json file to see what benchmarks are available.
        Incomplete list of supported benchmarks:
        - 'cartpolereduced', 'cartpolefull',
        - 'xgboost',
        - 'learna', 'metalearna',
        - 'NASCifar10ABenchmark', 'NASCifar10BBenchmark', 'NASCifar10CBenchmark',
        - 'SliceLocalizationBenchmark', 'ProteinStructureBenchmark',
          'NavalPropulsionBenchmark', 'ParkinsonsTelemonitoringBenchmark'

    output_dir : str, Path
        The optimizer stored in the previous optimization run its results in this directory. The scheduler looks for
        trajectory files, validated runhistories and unvalidated runhistories.

    procedure : str
        Either 'start_worker' or 'start_scheduler'.
        If 'start_scheduler' is selected, then a nameserver, a scheduler, and a worker is started. This is equal to
            run the process on a local machine.
        By selecting 'start_worker', only a worker is started. The worker then starts to look for the nameserver address
            in the output_directory and connects to the scheduler. By starting additional worker, you can distribute the
            validation process to multiple machines.

    run_id : str
        Unique name of the validation run

    worker_id : int

    rng : Optional[int]
        Random seed for the experiment. This seed is passed to the benchmark. By default 0.

    use_local : Optional[bool]
        If you want to use the HPOBench benchmarks in a non-containerized version (installed locally inside the
        current python environment), you can set this parameter to True. This is not recommend.

    debug : Optional[bool]
        Increase the verbosity level

    interface : Optional[str]
        The name of the network interface, where the nameserver is about to start. Defaults to 'lo' (localhost).
        This option is only required, if you want to start the scheduler. (The parameter `procedure` is set to
        'start_scheduler').

    nameserver_port : Optional[int]
        Start the nameserver serving on this port. Defaults to 0. If the port is 0, a random, free port is automatically
        selected (recommend).

    recompute_all : bool, None
        If you want to recompute all validation results regardless of whether already precomputed results exist or not.

    benchmark_params : Dict
        Some benchmarks take special parameters for the initialization. For example, The XGBOostBenchmark takes as
        input a task_id. This task_id specifies the OpenML dataset to use.

        Please take a look into the HPOBench Benchmarks to find out if the benchmark needs further parameter.
        Note: Most of them don't need further parameters.
    """

    _root_log.info(f'Start validating procedure on benchmark {benchmark}')

    if debug:
        _root_log.setLevel(level=logging.DEBUG)
        main_logger.setLevel(level=logging.DEBUG)
        from hpobench.util.container_utils import enable_container_debug
        enable_container_debug()

    output_dir = Path(output_dir)
    assert output_dir.is_dir(), f'Result folder doesn\'t exist: {output_dir}'

    benchmark_settings = get_benchmark_settings(benchmark)
    credentials_file = output_dir / f'HPOBenchExpUtils_nameserver_{run_id}.json'

    current_ip = nic_name_to_host(interface)
    _root_log.info(f'ValidateBenchmark: Current IP: {current_ip}')

    # If we only like to have a single worker:
    if procedure == 'start_worker':
        from HPOBenchExperimentUtils.core.worker import start_worker
        start_worker(benchmark=benchmark, credentials_dir=output_dir, run_id=run_id, worker_id=worker_id,
                     worker_ip_address=current_ip, rng=rng, use_local=use_local, debug=debug, **benchmark_params)
        return 1

    # If the old credentials file exists, we have to delete it. Otherwise a worker could connect to an old address and
    # throw an error.
    if credentials_file.exists():
        credentials_file.unlink()

    # Otherwise we want to start a scheduler. The scheduler reads in all the configurations and
    # distributes them to the worker.
    # We also start a worker in the background, since the scheduler is pretty cheap.
    if procedure != 'start_scheduler':
        raise ValueError(f'Procedure must be either start_scheduler or start_worker, but was {procedure}')

    # Start a nameserver in the background
    ns_ip = current_ip

    # STEP 1: Load the configuration which should be validated.
    # Find the paths to the trajectory files
    trajectories_paths = list(output_dir.rglob(TRAJECTORY_V1_FILENAME))
    trajectories_paths += list(output_dir.rglob(TRAJECTORY_V2_FILENAME))
    trajectories_paths += list(output_dir.rglob(TRAJECTORY_V3_FILENAME))

    # Load both trajectories: The larger-is-better-trajectory (v1) and the only-better-counts-trajectory
    trajectories = load_json_files(trajectories_paths)

    # Extract the configurations which should be validated from the both trajectories and combine the results.
    configurations = extract_configs_from_trajectories(trajectories)

    # Load the results (already validated configurations) from previous runs
    validated_runhistory_paths = list(output_dir.rglob(VALIDATED_RUNHISTORY_FILENAME))
    already_evaluated_configs = load_configs_with_function_values_from_runhistories(validated_runhistory_paths)

    # This dict stores all results from the validation procedure (key is the configuration but as str)
    validation_results = {}

    # Memorize all configurations (as dicts)
    unvalidated_configurations_total = []

    # Store the configurations (as dicts) for which we dont have already results from previous runs.
    unvalidated_configurations = []

    # Get the actual configurations to validate (Remove duplicates)
    for config in configurations:
        config_str = str(config)

        # The configuration is not unique and we have already seen it
        if config_str in validation_results:
            continue

        validation_results[str(config)] = None
        unvalidated_configurations_total.append(config)

        # We have not validated this configuration previously
        if config_str not in already_evaluated_configs:
            unvalidated_configurations.append(config)

    main_logger.info('Finished loading unvalidated and already validated configurations')
    main_logger.info(f'Found {len(trajectories_paths)} trajectories with a total of {len(configurations)} configs. '
                     f'(Unique: {len(validation_results)})')
    main_logger.info(f'Found {len(already_evaluated_configs)} already validated configurations.')

    if recompute_all:
        main_logger.info('Recompute all configurations.')
        unvalidated_configurations = unvalidated_configurations_total
    else:
        validation_results.update(already_evaluated_configs)
    main_logger.info(f'Going to evaluate {len(unvalidated_configurations)} configurations.')

    if len(unvalidated_configurations) == 0:
        main_logger.info(f'There are no configurations to evaluate. Stop this procedure')
        return 0

    main_logger.info(f'Going to start the nameserver on {ns_ip}:{nameserver_port}')
    ns_ip, ns_port = start_nameserver(ns_ip=ns_ip, ns_port=nameserver_port,
                                      credentials_file=credentials_file,
                                      thread_name=f'HPOBenchExpUtils Run {run_id}')

    with Worker(run_id=run_id, worker_id=worker_id, ns_ip=ns_ip, ns_port=ns_port, object_ip=current_ip,
                debug=debug) as worker:

        worker.start_up(benchmark_settings, benchmark_params, rng, use_local)
        main_logger.debug(f'Benchmark initialized. Additional benchmark parameters {benchmark_params}')
        worker_thread = threading.Thread(target=worker.run, name=f'Worker Thread {worker_id}', daemon=True)
        worker_thread.start()

        # give the scheduler the list of configs to validate
        fidelity_space = worker.benchmark.get_fidelity_space()
        default_fidelity = fidelity_space.get_default_configuration()
        default_fidelity = default_fidelity.get_dictionary()

        contents = [Content(config, default_fidelity, benchmark_params) for config in unvalidated_configurations]

        with Scheduler(run_id=run_id, ns_ip=ns_ip, ns_port=ns_port, object_ip=current_ip,
                       output_dir=output_dir, contents=contents) as scheduler:

            main_logger.info('Start the Scheduler')
            main_logger.info(f'Going to validate {len(contents)} configuration.')

            scheduler.run()
            results = scheduler.get_results_by_configuration()

            main_logger.info('Scheduler has finished.')

    validation_results.update(results)

    # Go through all found unvalidated trajectories and create for each of these files a new trajectory whith the
    # validated results.
    for unvalidated_traj, unvalidated_traj_path in zip(trajectories, trajectories_paths):
        write_validated_trajectory(unvalidated_traj, validation_results, unvalidated_traj_path)

    main_logger.info('Validating the trajectory was successful.')
    return 1


def parse_args():
    main_parser = argparse.ArgumentParser(description='HPOBench validated a trajectory from a benchmark with a '
                                                      'unified interface')

    subparsers = main_parser.add_subparsers(title='Available functions',
                                            dest='procedure')

    # COMMON ARGUMENTS
    common_args_parser = argparse.ArgumentParser(add_help=False)
    common_args_parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    common_args_parser.add_argument('--output_dir', type=str, required=True,
                                    help='The unvalidated trajectories are here. Also the scheduler writes the '
                                         'validated results into this directory, too.')
    common_args_parser.add_argument('--run_id', type=str, required=True, help='Unique name of the run')
    common_args_parser.add_argument('--worker_id', type=int, required=True, help='Unique name of the worker')
    common_args_parser.add_argument('--interface', type=str, required=True,
                                    help='Name of the interface, where the machines can communicate. '
                                         'E.g. lo for localhost')
    common_args_parser.add_argument('--rng', required=False, default=0, type=int)
    common_args_parser.add_argument('--use_local', action='store_true', default=False)
    common_args_parser.add_argument('--debug', action='store_true', default=False,
                                    help="When given, enables debug mode logging.")

    # Now, specify if the scheduler should start or just a single worker.
    scheduler_parser = subparsers.add_parser('start_scheduler', help='Start the Scheduler, Nameserver and a Worker',
                                             parents=[common_args_parser])
    scheduler_parser.add_argument('--nameserver_port', type=int, default=0, required=False)
    scheduler_parser.add_argument('--recompute_all', action='store_true', default=False)

    worker_parser = subparsers.add_parser('start_worker', help='Start only a worker.',
                                          parents=[common_args_parser])

    args, unknown = main_parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    args, unknown = parse_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)
    validate_benchmark(**vars(args), **benchmark_params)
