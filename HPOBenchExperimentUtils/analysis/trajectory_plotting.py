import logging
from pathlib import Path
from typing import Union

from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc
from HPOBenchExperimentUtils import _default_log_format, _log as _main_log
from HPOBenchExperimentUtils.utils.validation_utils import load_trajectories, load_trajectories_as_df,\
    get_statistics_df, df_per_optimizer
from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_setting

import matplotlib.pyplot as plt

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


def read_trajectories(benchmark: str, input_dir: Path, train: bool=True, y_best: float=0.0):
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    unique_optimizer = load_trajectories_as_df(input_dir=input_dir,
                                               which="train" if train else "test")
    optimizer_names = list(unique_optimizer.keys())
    statistics_df = []
    _log.critical("Found: " + ",".join(optimizer_names))
    if len(optimizer_names) == 0:
        raise ValueError("No files found")
    for key in optimizer_names:
        trajectories = load_trajectories(unique_optimizer[key])
        optimizer_df = df_per_optimizer(key, trajectories, y_best=y_best)
        statistics_df.append(get_statistics_df(optimizer_df))
    return optimizer_names, trajectories, statistics_df


def plot_trajectory(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str],
                    criterion: str = 'mean', unvalidated: bool = True, **kwargs):
    _log.info(f'Start plotting trajectories of benchmark {benchmark}')
    benchmark_spec = plot_dc.get(benchmark, {})
    y_best = benchmark_spec.get("ystar_valid", 0) if unvalidated \
        else benchmark_spec.get("ystar_test", 0)

    optimizer_names, trajectories, statistics_df = read_trajectories(
        benchmark=benchmark,
        input_dir=Path(input_dir),
        train=unvalidated,
        y_best=y_best,
    )
    # start plotting the trajectories:
    f, ax = plt.subplots(1, 1)
    min_ = 100000
    max_ = -1
    for key, df in zip(optimizer_names, statistics_df):
        label = get_optimizer_setting(key).get("display_name", key)

        df[criterion].plot.line(drawstyle='steps-post', linewidth=2, ax=ax, label=label)
        min_ = min(min_, df[criterion].min())
        max_ = max(max_, df['q25'].max())
        if criterion == 'mean':
            ax.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.3, step="post")
            min_ = min(min_, df[criterion].min())
            max_ = max(max_, df[criterion].max())
        else:
            ax.fill_between(df.index, df['q25'], df['q75'], alpha=0.3, step="post")
            min_ = min(min_, df['q25'].min())
            max_ = max(max_, df[criterion].max())

    if y_best != 0:
        ylabel = "Regret"
    else:
        ylabel = "Loss"
    xl, xu = ax.get_xlim()

    xl = benchmark_spec.get("xlim_lo", 1)
    xu = benchmark_spec.get("xlim_up", xu)
    yl = benchmark_spec.get("ylim_lo", min_)
    yu = benchmark_spec.get("ylim_up", max_)
    xscale = benchmark_spec.get("xscale", "log")
    yscale = benchmark_spec.get("yscale", "log")
    xlabel = benchmark_spec.get("xlabel", "Runtime in seconds")
    ylabel = benchmark_spec.get("xlabel", "Loss")
    test_str = 'Train' if unvalidated else 'Test'
    ylabel = f'{criterion.capitalize()} {test_str} {ylabel}'

    ax.set_xlim([xl, xu])
    ax.set_ylim([yl, yu])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()
    val_str = '' if unvalidated else 'validated'
    ax.set_title(f'{benchmark}')
    plt.grid(b=True, which="both", axis="both")
    plt.savefig(Path(output_dir) / f'{benchmark}_{val_str}_{criterion}_trajectory.png')
    return 1
