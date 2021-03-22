import logging
from pathlib import Path
from typing import Union, List

from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc, color_per_opt, unify_layout
from HPOBenchExperimentUtils import _log as _main_log
from HPOBenchExperimentUtils.utils.validation_utils import load_json_files, load_trajectories_as_df
from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_setting, get_benchmark_settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)


def read_trajectories(benchmark: str, input_dir: Path, train: bool=True,
                      which: str="v1", opt_list: Union[List[str], None] = None):
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

    trajectories = []
    for key in opt_list:
        if key not in optimizer_names:
            raise ValueError(f"Not the same opts for all benchmarks; {benchmark} missed {key}")
        trs = load_json_files(unique_optimizer[key])
        series_list = []
        for t in trs:
            times = np.array([r["total_time_used"] for r in t[1:]])
            vals = np.array([r["function_value"] for r in t[1:]])
            series_list.append(pd.Series(data=vals, index=times))
        series = pd.concat(series_list, axis=1)
        # Fill missing performance values (NaNs) with last non-NaN value.
        series = series.fillna(method='ffill')

        vali = -1
        for c in series.columns:
            vali = max(vali, series[c].first_valid_index())
        series = series.loc[vali:]
        trajectories.append(series)
    return trajectories


def plot_ranks(benchmarks: List[str], output_dir: Union[Path, str], input_dir: Union[Path, str],
               criterion: str = 'mean', unvalidated: bool = True, which: str = "v1",
               opt_list: Union[List[str], None] = None,  **kwargs):
    _log.info(f'Start plotting ranks of benchmarks {benchmarks}')

    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    benchmark_spec = plot_dc.get(benchmarks[0], {})
    benchmark_settings = get_benchmark_settings(benchmarks[0])

    all_trajectories = []
    horizon = []
    for b in benchmarks:
        benchmark_settings = get_benchmark_settings(b)
        horizon.append(benchmark_settings['time_limit_in_s'])
        tr = read_trajectories(benchmark=b, input_dir=input_dir, train=unvalidated, which=which,
                               opt_list=opt_list)
        all_trajectories.append(tr)
    assert len(set(horizon)) == 1
    horizon = int(horizon[0])
    _log.info(f"Handling horizon: {horizon}sec")

    # Step 2. Compute average ranks of the trajectories.
    #####################################################################################
    all_rankings = []
    n_tasks = len(benchmarks)
    paired = False

    n_iter = 500
    if paired:
        n_iter = all_trajectories[0][0].shape[1]

    for i in range(n_iter):
        if i % 50 == 0: print("%d / %d" % (i, n_iter))
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
                 c=color_per_opt.get(key, "k"),
                 linewidth=2)
    if benchmark_settings["is_surrogate"]:
        plt.xlabel("Simulated runtime in seconds")
    else:
        plt.xlabel("Runtime in seconds")
    ax.set_ylabel(f"{criterion.capitalize()} rank")
    ax.set_ylim([0.9, len(opt_list)+0.1])
    ax.set_xlim(10, horizon)
    ax.set_xscale(benchmark_spec.get("xscale", "log"))

    unify_layout(ax, title=",".join(benchmarks))
    val_str = 'optimized' if unvalidated else 'validated'
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'ranks_{".".join(benchmarks)}_{val_str}_{which}.png')
    plt.close('all')
    return 1