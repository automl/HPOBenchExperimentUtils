import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from HPOlibExperimentUtils import _default_log_format, _log as _main_log
from HPOlibExperimentUtils.utils.validation_utils import load_trajectories, \
    load_trajectories_as_df, df_per_optimizer
from HPOlibExperimentUtils.utils.plotting_utils import color_per_opt
from HPOlibExperimentUtils.utils.runner_utils import get_optimizer_setting

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


def plot_fidels(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Plotting evaluated fidelities of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")

    stat_dc = {}
    for opt in opt_rh_dc:
        if len(opt_rh_dc) == 0: continue
        rhs = load_trajectories(opt_rh_dc[opt])
        df = df_per_optimizer(opt, rhs)
        stat_dc[opt] = df

    plt.figure(figsize=[10, 5])
    a = plt.subplot(111)
    for opt in stat_dc:
        # get queried fidels
        df = stat_dc[opt]
        nseeds = df['id'].unique()
        for seed in nseeds:
            fidels = df[df['id'] == seed]["fidelity_value"]
            steps = df[df['id'] == seed]["total_objective_costs"]
            label = get_optimizer_setting(opt).get("display_name", opt)

            a.scatter(steps, fidels, edgecolor=color_per_opt.get(opt, "k"), facecolor="none",
                      marker="o", label=label if seed == 0 else None)
    a.legend()
    a.set_xscale("log")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{benchmark}_fidel.png')


def plot_overhead(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Start plotting overhead of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")

    stat_dc = {}
    for opt in opt_rh_dc:
        if len(opt_rh_dc) == 0: continue
        rhs = load_trajectories(opt_rh_dc[opt])
        df = df_per_optimizer(opt, rhs)
        stat_dc[opt] = df

    plt.figure(figsize=[5, 5])
    a = plt.subplot(111)
    for opt in stat_dc:
        # get queried fidels
        df = stat_dc[opt]
        nseeds = df['id'].unique()
        for seed in nseeds:
            steps = df[df['id'] == seed]["total_objective_costs"]
            overhead = df[df['id'] == seed]["start_time"] - df[df['id'] == seed]["finish_time"].shift(1)

            overhead = np.cumsum(overhead)
            label = get_optimizer_setting(opt).get("display_name", opt)
            plt.plot(steps, overhead, color=color_per_opt.get(opt, "k"), label=label if seed == 0 else None)

            benchmark_cost = df[df['id'] == seed]["finish_time"] - df[df['id'] == seed]["start_time"]
            benchmark_cost = np.cumsum(benchmark_cost)
            plt.plot(steps, benchmark_cost, color='k', alpha=0.5, zorder=0,
                     label=benchmark if seed == 0 and opt == list(stat_dc.keys())[0] else None)

    a.legend()
    a.grid()
    a.set_yscale("log")
    a.set_xscale("log")
    a.set_xlabel("Runtime in seconds")
    a.set_ylabel("Cumulated overhead in seconds")
    a.set_xlim([1, a.set_xlim()[1]])
    a.set_ylim([0, 1000])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{benchmark}_overhead.png')

