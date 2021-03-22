import logging
from pathlib import Path
from typing import Union, List

from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc, color_per_opt, unify_layout
from HPOBenchExperimentUtils import _log as _main_log
from HPOBenchExperimentUtils.utils.validation_utils import load_json_files, load_trajectories_as_df,\
    get_statistics_df, df_per_optimizer
from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_setting, get_benchmark_settings

import matplotlib.pyplot as plt

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)


def read_trajectories(benchmark: str, input_dir: Path, train: bool=True, y_best: float=0.0,
                      which: str="v1", opt_list: Union[List[str], None] = None):
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    unique_optimizer = load_trajectories_as_df(
        input_dir=input_dir, which=f'train_{which}' if train else f'test_{which}')
    if opt_list is None:
        opt_list = list(unique_optimizer.keys())
    statistics_df = []
    _log.critical("Found: " + ",".join(list(unique_optimizer.keys())))
    if len(unique_optimizer) == 0:
        raise ValueError("No files found")
    optimizer_names = []
    for key in unique_optimizer.keys():
        if key not in opt_list:
            _log.info(f'Skip {key}')
            continue
        optimizer_names.append(key)
        trajectories = load_json_files(unique_optimizer[key])
        optimizer_df = df_per_optimizer(key, trajectories, y_best=y_best)
        statistics_df.append(get_statistics_df(optimizer_df))
    return optimizer_names, trajectories, statistics_df


def plot_trajectory(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str],
                    criterion: str = 'mean', unvalidated: bool = True, which: str = "v1",
                    opt_list: Union[List[str], None] = None,  **kwargs):
    _log.info(f'Start plotting trajectories of benchmark {benchmark}')

    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    benchmark_spec = plot_dc.get(benchmark, {})
    benchmark_settings = get_benchmark_settings(benchmark)

    y_best = benchmark_spec.get("ystar_valid", 0) if unvalidated \
        else benchmark_spec.get("ystar_test", 0)

    optimizer_names, trajectories, statistics_df = read_trajectories(
        benchmark=benchmark,
        input_dir=Path(input_dir),
        train=unvalidated,
        y_best=y_best,
        which=which,
        opt_list=opt_list,
    )
    # start plotting the trajectories:
    f, ax = plt.subplots(1, 1)
    min_ = 100000
    max_ = -1
    for key, df in zip(optimizer_names, statistics_df):
        try:
            label = get_optimizer_setting(key).get("display_name", key)
        except:
            _log.critical(f'Skip unknown optimizer {key}')
            continue
        color = color_per_opt.get(key, "k")
        df[criterion].plot.line(drawstyle='steps-post', linewidth=2, ax=ax, label=label, c=color)
        min_ = min(min_, df[criterion].min())
        max_ = max(max_, df['q25'].max())
        
        if criterion == 'mean':
            ax.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.3, step="post", facecolor=color)
            min_ = min(min_, df[criterion].min())
            max_ = max(max_, df[criterion].max())
        else:
            ax.fill_between(df.index, df['q25'], df['q75'], alpha=0.3, step="post", facecolor=color)
            min_ = min(min_, df['q25'].min())
            max_ = max(max_, df[criterion].max())

    # TODO: This statement has no effect. Overwritten in line 82 without usage
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
    if benchmark_settings["is_surrogate"] == True:
        xlabel = benchmark_spec.get("xlabel", "Simulated runtime in seconds")
    else:
        xlabel = benchmark_spec.get("xlabel", "Runtime in seconds")
    test_str = 'Optimized' if unvalidated else 'Test'
    ylabel = f'{criterion.capitalize()} {test_str} {ylabel}'

    ax.set_xlim([xl, xu])
    ax.set_ylim([yl, yu])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    val_str = 'optimized' if unvalidated else 'validated'

    unify_layout(ax, title=f'{benchmark}')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'trajectory_{benchmark}_{val_str}_{criterion}_{which}.png')
    plt.close('all')
    return 1
