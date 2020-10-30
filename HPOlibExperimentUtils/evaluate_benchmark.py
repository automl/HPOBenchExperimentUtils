import argparse
import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hpolib.util.example_utils import set_env_variables_to_use_only_one_core

from HPOlibExperimentUtils.utils.runner_utils import get_benchmark_names
from HPOlibExperimentUtils.utils.validation_utils import load_trajectories, load_trajectories_as_df, get_statistics_df, \
    df_per_optimizer

from HPOlibExperimentUtils import _default_log_format, _log as _main_log

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)

set_env_variables_to_use_only_one_core()

plot_dc = {
    "BNNOnBostonHousing": {
    # BOHB paper
    "xlim_lo": 10**3,
    "ylim_lo": 3,
    "ylim_up": 9,
    "xscale": "log",
    "yscale": "linear",
    },
    "BNNOnProteinStructure": {
    "xlim_lo": 10**3,
    "ylim_lo": 2,
    "ylim_up": 10,
    "xscale": "log",
    "yscale": "linear",
    },
    "BNNOnYearPrediction": {
    "xlim_lo": 10**3,
    "ylim_lo": 10,
    "ylim_up": 50,
    "xscale": "log",
    "yscale": "linear",
    },
    "cartpolereduces": {
    "xlim_lo": 10**1,
    "ylim_lo": 10**2,
    "ylim_up": 10**4,
    "xscale": "log",
    "yscale": "log",
    },
}

def save_table(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str],
               unvalidated: bool = True, **kwargs):
    _log.info(f'Start creating table of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {output_dir}'
    unique_optimizer, val_str = load_trajectories_as_df(input_dir, unvalidated)

    keys = list(unique_optimizer.keys())
    result_df = pd.DataFrame()
    for key in keys:
        trajectories = load_trajectories(unique_optimizer[key])
        optimizer_df = df_per_optimizer(key, trajectories)

        unique_ids = np.unique(optimizer_df['id'])
        for unique_id in unique_ids:
            df = optimizer_df[optimizer_df['id'] == unique_id]
            df = df.sort_values(by='total_time_used')
            last_inc = df.tail(1)
            result_df = result_df.append(last_inc)

    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    # q1 = lambda x: x.quantile(0.25)
    # q3 = lambda x: x.quantile(0.75)
    aggregate_funcs = ['mean', 'std', 'median', q1, q3, 'min', 'max']

    result_df = result_df.groupby('optimizer').agg({'function_values': aggregate_funcs,
                                                    'total_time_used': aggregate_funcs})

    result_df.columns = ["_".join(x) for x in result_df.columns.ravel()]

    val_str = 'unvalidated' if unvalidated else 'validated'
    with open(Path(output_dir) / f'{benchmark}_{val_str}_result_table.tex', 'w') as fh:
        fh.write(result_df.to_latex())


def plot_trajectory(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str],
                    criterion: str = 'mean', unvalidated: bool = True, **kwargs):

    _log.info(f'Start plotting trajectories of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {output_dir}'

    unique_optimizer, val_str = load_trajectories_as_df(input_dir, unvalidated)

    keys = list(unique_optimizer.keys())
    statistics_df = []
    for key in keys:
        trajectories = load_trajectories(unique_optimizer[key])
        optimizer_df = df_per_optimizer(key, trajectories)
        statistics_df.append(get_statistics_df(optimizer_df))

    # start plotting the trajectories:
    f, ax = plt.subplots(1, 1)
    min_ = 100000
    max_ = -1
    for key, df in zip(keys, statistics_df):

        df[criterion].plot.line(drawstyle='steps-post', linewidth=2, ax=ax, label=key)
        min_ = min(min_, df[criterion].min())
        max_ = max(max_, df[criterion].max())
        if criterion == 'mean':
            ax.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.3)
        else:
            ax.fill_between(df.index, df['q25'], df['q75'], alpha=0.3)

    ax.set_ylabel('%s Loss' % criterion)
    xl, xu = ax.get_xlim()
    
    benchmark_spec = plot_dc.get(benchmark, {})
    xl = benchmark_spec.get("xlim_lo", xl)
    xu = benchmark_spec.get("xlim_up", xu)
    yl = benchmark_spec.get("ylim_lo", min_)
    yu = benchmark_spec.get("ylim_up", max_)
    xscale = benchmark_spec.get("xscale", "log")
    yscale = benchmark_spec.get("yscale", "log")

    ax.set_xlim([xl, xu])
    ax.set_ylim([yl, yu])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(f'{benchmark} {val_str} {criterion}')
    ax.legend()
    val_str = 'unvalidated' if unvalidated else 'validated'
    plt.savefig(Path(output_dir) / f'{benchmark}_{val_str}_{criterion}_trajectory.png')
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper - Plotting tool',
                                     description='Plot the trajectories')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--what', choices=["all", "table", "over_time"], default="all")
    parser.add_argument('--agg', choices=["mean", "median"], default="median")
    parser.add_argument('--unvalidated', action='store_true', default=False)

    args, unknown = parser.parse_known_args()

    if args.what in ("all", "table"):
        save_table(**vars(args))

    if args.what in ("all", "over_time"):
        plot_trajectory(criterion=args.agg, **vars(args))
