import argparse
import logging

from HPOlibExperimentUtils.utils.runner_utils import get_benchmark_names
from HPOlibExperimentUtils.analysis.trajectory_plotting import plot_trajectory
from HPOlibExperimentUtils.analysis.stats_generation import plot_fidels, plot_overhead
from HPOlibExperimentUtils.analysis.table_generation import save_table
from HPOlibExperimentUtils import _default_log_format, _log as _main_log

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper - Plotting tool',
                                     description='Plot the trajectories')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--what', choices=["all", "table", "over_time", "other"], default="all")
    parser.add_argument('--agg', choices=["mean", "median"], default="median")
    parser.add_argument('--unvalidated', action='store_true', default=False)

    args, unknown = parser.parse_known_args()

    if args.what in ("all", "table"):
        save_table(**vars(args))

    if args.what in ("all", "over_time"):
        plot_trajectory(criterion=args.agg, **vars(args))

    if args.what in ("all", "other"):
        plot_fidels(**vars(args))
        plot_overhead(**vars(args))

