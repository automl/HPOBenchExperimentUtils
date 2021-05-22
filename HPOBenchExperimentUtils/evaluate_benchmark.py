import argparse
import logging
import sys

from HPOBenchExperimentUtils.utils.runner_utils import get_benchmark_names
from HPOBenchExperimentUtils.analysis.trajectory_plotting import plot_trajectory
from HPOBenchExperimentUtils.analysis.stats_generation import plot_fidels, plot_overhead, \
    plot_ecdf, plot_correlation, get_stats
from HPOBenchExperimentUtils.analysis.table_generation import save_median_table
from HPOBenchExperimentUtils.analysis.rank_plotting import plot_ranks, plot_ecdf_per_family
from HPOBenchExperimentUtils import _log as _root_log
from HPOBenchExperimentUtils.utils.plotting_utils import benchmark_families

_root_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HPOBench Wrapper - Plotting tool',
                                     description='Plot the trajectories')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), type=str, default=None)
    parser.add_argument('--what', choices=["all", "best_found", "over_time", "other",
                                           "ecdf", "correlation", "stats"], default="all")
    parser.add_argument('--rank', choices=benchmark_families.keys(), default=None)
    parser.add_argument('--agg', choices=["mean", "median"], default="median")
    parser.add_argument('--unvalidated', action='store_true', default=False)
    parser.add_argument('--which', choices=["v1", "v2"], default="v1")
    args, unknown = parser.parse_known_args()

    if args.unvalidated:
        list_of_opt_to_consider = ["autogluon",
                                   "dragonfly_default", 
                                   "randomsearch",
                                   "smac_sf", 
                                   "smac_hb_eta_3",
                                   "smac_bo",
                                   "dehb", 
                                   "hpbandster_bohb_eta_3",
                                   "hpbandster_hb_eta_3",
                                   "hpbandster_tpe",
                                   "de"
                                   #"mumbo",
                                   ]
    else:
        list_of_opt_to_consider = ["autogluon",
                                   "dragonfly_default", 
                                   "randomsearch",
                                   "smac_sf", 
                                   "smac_hb_eta_3",
                                   "smac_bo",
                                   "dehb", 
                                   "hpbandster_bohb_eta_3",
                                   "hpbandster_hb_eta_3",
                                   "hpbandster_tpe",
                                   "de"
                                   #"mumbo",
                                   ]

    if args.rank is None:
        assert args.benchmark is not None, f"If rank={args.rank}, then --benchmark must be set"
    else:
        _log.info("Only plotting metrics per family")
        #if args.unvalidated is True:
        #    plot_ecdf_per_family(**vars(args), benchmarks=benchmark_families[args.rank], familyname=args.rank,
        #               opt_list=list_of_opt_to_consider)
        #else:
        #    _log.info("Skipping ECDF per family")
        plot_ranks(**vars(args), benchmarks=benchmark_families[args.rank], familyname=args.rank,
                   opt_list=list_of_opt_to_consider)
        
        sys.exit(1)


    if args.what in ("all", "best_found"):               
        save_median_table(**vars(args), opt_list=list_of_opt_to_consider, thresh=0.01)
        save_median_table(**vars(args), opt_list=list_of_opt_to_consider, thresh=0.1)
        save_median_table(**vars(args), opt_list=list_of_opt_to_consider, thresh=0.5)
        save_median_table(**vars(args), opt_list=list_of_opt_to_consider, thresh=1.0)

    if args.what in ("all", "over_time"):
        plot_trajectory(criterion=args.agg, **vars(args), opt_list=list_of_opt_to_consider,
                        whatobj='total_time_used')
        plot_trajectory(criterion=args.agg, **vars(args), opt_list=list_of_opt_to_consider,
                        whatobj='total_objective_costs')

    if args.what in ("all", "ecdf"):
        plot_ecdf(**vars(args), opt_list=list_of_opt_to_consider)

    if args.what in ("all", "correlation"):
        plot_correlation(**vars(args), opt_list=list_of_opt_to_consider)

    if args.what in ("all", "stats"):
        get_stats(**vars(args), opt_list=list_of_opt_to_consider)

    if args.what in ("all", "other"):
        if args.unvalidated is False:
            _log.critical("Statistics will be plotted on unvalidated data")
        plot_fidels(**vars(args), opt_list=list_of_opt_to_consider)
        plot_overhead(**vars(args), opt_list=list_of_opt_to_consider)
