"""
Script to generate the average ranking plots for the chosen optimizer families
"""

import pickle
import logging
import argparse
from pathlib import Path
from typing import Union, List

from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc, color_per_opt, unify_layout, \
    benchmark_dc, benchmark_families, linestyle_per_opt
from HPOBenchExperimentUtils import _log as _main_log
from HPOBenchExperimentUtils.utils.validation_utils import load_json_files, load_trajectories_as_df
from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_setting, get_benchmark_settings

from hpobench.benchmarks.ml import TabularBenchmark

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)


paper_tasks = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212, 168911,
    9981, 167120, 14965, 146606, 7592, 9977
]
ntasks_done = dict(
    svm=29,
    lr=29,
    rf=28,
    xgb=22,
    nn=8
)


def read_trajectories(benchmark: str, input_dir: Path, train: bool=True,
                      which: str="v1", opt_list: Union[List[str], None] = None, **kwargs):
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    unique_optimizer = load_trajectories_as_df(
        input_dir=input_dir, which=f'train_{which}' if train else f'test_{which}')
    optimizer_names = list(unique_optimizer.keys())
    if opt_list is None:
        opt_list = optimizer_names
    _log.critical("Found: " + ",".join(optimizer_names))
    if len(optimizer_names) == 0:
        raise ValueError("No files found")

    if "benchmark_spec" in kwargs:
        y_max = kwargs["benchmark_spec"]["y_max"]
        ystar_valid = kwargs["benchmark_spec"]["ystar_valid"]
    else:
        y_max = 1
        ystar_valid = 0
    normalizer = y_max - ystar_valid

    budget = 1 if "benchmark_settings" not in kwargs else \
        kwargs["benchmark_settings"]["time_limit_in_s"]

    trajectories = []
    # trajectories = dict()
    for key in opt_list:
        if key not in optimizer_names:
            raise ValueError(f"Not the same opts for all benchmarks; {benchmark} missed {key}")
        trs = load_json_files(unique_optimizer[key])
        series_list = []
        for t in trs:
            times = np.array([r["total_time_used"]/budget for r in t[1:]])
            vals = np.array([(r["function_value"] - ystar_valid) / normalizer for r in t[1:]])
            series_list.append(pd.Series(data=vals, index=times))
        series = pd.concat(series_list, axis=1)
        # Fill missing performance values (NaNs) with last non-NaN value.
        series = series.fillna(method='ffill')

        vali = -1
        for c in series.columns:
            try:
                vali = max(vali, series[c].first_valid_index())
            except:
                # print(series)
                print(c, key)
                # print(series[c].first_valid_index())
        series = series.loc[vali:]

        # if normalize_times_by != 1:
        # Here we assume max timestep is 1 and we discretize
        # because we're handling a large ranking plot
        steps = 10**np.linspace(-6, 0, 200)
        series = series.transpose()
        # Get data from csv
        for step in steps:
            if step > 0:
                series[step] = np.NaN
        a = series.sort_index(axis=1).ffill(axis=1)
        a = a.loc[:,steps]
        series = a
        series = series.transpose()

        trajectories.append(series)
        # trajectories[key] = series
    return trajectories


def plot_ranks(benchmarks: List[str], familyname: str, output_dir: Union[Path, str],
               input_dir: Union[Path, str], opts: str, criterion: str = 'mean',
               unvalidated: bool = True,
               which: str = "v1", opt_list: Union[List[str], None] = None, **kwargs):
    _log.info(f'Start plotting ranks of benchmarks {benchmarks}')

    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    benchmark_spec = plot_dc.get(benchmarks[0], {})
    benchmark_settings = get_benchmark_settings(benchmarks[0])

    all_trajectories = []
    horizon = []
    x_lo = []
    for b in benchmarks:
        print("Benchmark {}".format(b))
        benchmark_settings = get_benchmark_settings(b)
        benchmark_spec = plot_dc.get(b, {})
        x_lo.append(benchmark_spec.get("xlim_lo", 1))
        tr = read_trajectories(benchmark=b, input_dir=input_dir, train=unvalidated, which=which,
                               opt_list=opt_list, benchmark_spec=benchmark_spec,
                               benchmark_settings=benchmark_settings)
        # tr /= benchmark_settings['time_limit_in_s']
        horizon.append(1)  # benchmark_settings['time_limit_in_s'])
        all_trajectories.append(tr)
    assert len(set(horizon)) == 1
    horizon = int(horizon[0])
    _log.info(f"Handling horizon: {horizon}sec")
    x_lo = np.min(x_lo)

    # Step 2. Compute average ranks of the trajectories.
    #####################################################################################
    all_rankings = []
    n_tasks = len(benchmarks)  # len(benchmarks)
    paired = False

    n_iter = 5000
    if paired:
        n_iter = all_trajectories[0][0].shape[1]

    for i in range(n_iter):
        if i % 500 == 0: print("%d / %d" % (i, n_iter))
        if paired:
            pick = np.ones(len(opt_list), dtype=np.int) * i
        else:
            pick = np.random.choice(all_trajectories[0][0].shape[1],
                                    size=(len(opt_list)))
        for j in range(n_tasks):
            all_trajectories_tmp = pd.DataFrame(
                {opt_list[k]: at.iloc[:, pick[k]] for
                 k, at in enumerate(all_trajectories[j])}
            )
            all_trajectories_tmp = all_trajectories_tmp.fillna(method='ffill', axis=0)
            # bottom: assign highest rank to NaN values if ascending
            r_tmp = all_trajectories_tmp.rank(axis=1, na_option="bottom")
            all_rankings.append(r_tmp)

    final_ranks = []
    for i, model in enumerate(opt_list):
        print("{:>2}".format(i), end="\r")
        ranks_for_model = []
        for ranking in all_rankings:
            ranks_for_model.append(ranking.loc[:, model])
        ranks_for_model = pd.DataFrame(ranks_for_model)
        ranks_for_model = ranks_for_model.fillna(method='ffill', axis=1)
        if criterion == "mean":
            final_ranks.append(ranks_for_model.mean(skipna=True))
        elif criterion == "median":
            final_ranks.append(ranks_for_model.median(skipna=True))

    # Step 3. Plot the average ranks over time.
    ######################################################################################
    f, ax = plt.subplots(1, 1)
    for i, key in enumerate(opt_list):
        X_data = []
        y_data = []
        for x, y in final_ranks[i].iteritems():
            X_data.append(x)
            y_data.append(y)
        X_data.append(horizon)
        X_data = np.array(X_data)
        y_data.append(y)
        plt.plot(X_data, y_data, label=get_optimizer_setting(key).get("display_name", key),
                 linestyle=linestyle_per_opt.get(key, 'solid'),
                 c=color_per_opt.get(key, "k"),
                 linewidth=2)
    if benchmark_settings["is_surrogate"]:
        plt.xlabel("Simulated runtime in seconds")
    else:
        plt.xlabel("Runtime in seconds")
        
    if familyname == "all":
        plt.xlabel("Fraction of budget")
        ax.set_xlim([kwargs["x_lo"], 1])
    else:
        ax.set_xlim([x_lo, horizon])

    ax.set_xscale(benchmark_spec.get("xscale", "log"))
    ax.set_ylabel(f"{criterion.capitalize()} rank")
    ax.set_ylim([0.9, len(opt_list) + 0.1])

    # unify_layout(ax, title=None, add_legend=False)
    # ax.set_xlim([x_lo, horizon])
    # ax.set_xscale(benchmark_spec.get("xscale", "log"))

    unify_layout(ax, title=None, add_legend=False)
    val_str = 'optimized' if unvalidated else 'validated'
    plt.tight_layout()
    if "name" in kwargs:
        name = kwargs["name"]
        filename = Path(output_dir) / \
                   f'{name}_ranks_tabular_{familyname}_{val_str}_{which}_{opts}_{args.fig_type}.png'
    else:
        filename = Path(output_dir) / \
                   f'all_ranks_tabular_{familyname}_{val_str}_{which}_{opts}_{args.fig_type}.png'
    print(filename)
    plt.savefig(filename)
    plt.close('all')
    return 1


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default="/work/dlclarge1/mallik-hpobench/opt-results/evals/ranks")
    parser.add_argument('--input_dir', type=str,
                        default="/work/dlclarge1/mallik-hpobench/opt-results/runs/")
    parser.add_argument('--what', choices=["all", "best_found", "over_time", "other",
                                           "ecdf", "correlation", "stats"], default="best_found")
    parser.add_argument('--rank', default="all")
    parser.add_argument('--agg', choices=["mean", "median"], default="median")
    parser.add_argument('--unvalidated', action='store_true', default=False)
    parser.add_argument('--which', choices=["v1", "v2"], default="v1")
    parser.add_argument('--opts', default="all")
    parser.add_argument('--tabular', choices=["svm", "lr", "rf", "xgb", "nn"], default=None)
    parser.add_argument('--fig_type', choices=list(opt_list.keys()), default="fig4_sf")
    parser.add_argument('--x_lo', type=float, default=10**-6)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    opt_list = dict()
    opt_list['main_sf'] = ['randomsearch', 'de', 'smac_bo', 'smac_sf', 'ray_hyperopt',
                           'hpbandster_tpe']
    opt_list['main_mf'] = ['hpbandster_hb_eta_3', 'hpbandster_bohb_eta_3', 'dehb',
                           'smac_hb_eta_3', 'dragonfly_default',
                           'ray_hyperopt_asha']  # no 'optuna_tpe_median' and 'optuna_tpe_hb'

    # lists for the main paper
    opt_list['table3'] = opt_list['main_sf'] + opt_list['main_mf']  # table
    opt_list['fig4_sf'] = opt_list['main_sf']  # ranking across all benchs
    opt_list['fig4_mf'] = opt_list['main_mf']  # ranking across all benchs
    opt_list['fig5bohb'] = ['randomsearch', 'hpbandster_tpe', 'hpbandster_hb_eta_3',
                            'hpbandster_bohb_eta_3']  # ranking across all benchs
    opt_list['fig5dehb'] = ['randomsearch', 'de', 'hpbandster_hb_eta_3',
                            'dehb']  # ranking across all benchs
    opt_list['fig5all'] = opt_list['main_sf'] + opt_list['main_mf']

    # lists for the appendix
    opt_list['all_sf'] = ['randomsearch', 'de', 'smac_bo', 'smac_sf', 'ray_hyperopt',
                          'hpbandster_tpe']  # table + trajectory per bench + ranking per bench
    opt_list['all_mf'] = ['hpbandster_hb_eta_3', 'hpbandster_bohb_eta_3', 'dehb', 'smac_hb_eta_3',
                          'dragonfly_default', 'ray_hyperopt_asha', 'optuna_tpe_median',
                          'optuna_tpe_hb']  # table + trajectory per bench + ranking per bench
    opt_list['all_all'] = opt_list['all_sf'] + opt_list['all_mf']

    args = input_args()

    list_of_opt_to_consider = opt_list[args.fig_type]
    if args.tabular is not None:
        benchmarks = benchmark_families["tabular_{}".format(args.tabular)]
        args.name = args.tabular
        args.tabular = [args.tabular]
    else:
        args.tabular = ["lr", "svm", "rf", "xgb", "nn"]
        args.name = "all"
        benchmarks = []
        for b in ["lr", "svm", "rf", "xgb", "nn"]:
            benchmarks.extend(benchmark_families["tabular_{}".format(b)])

    if "svm" in args.tabular and "optuna_tpe_hb" in list_of_opt_to_consider:
        list_of_opt_to_consider.remove("optuna_tpe_hb")

    def check_task_id(benchmark_name):
        model, tid = benchmark_name.split("_")
        if int(tid) in paper_tasks[:ntasks_done[model]]:
            return True
        return False

    benchmarks = [name for name in benchmarks if check_task_id(name)]

    plot_ranks(
        **vars(args), benchmarks=benchmarks, familyname=args.rank, opt_list=list_of_opt_to_consider
    )
