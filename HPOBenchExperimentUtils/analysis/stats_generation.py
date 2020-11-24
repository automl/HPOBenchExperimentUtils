import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from HPOBenchExperimentUtils import _default_log_format, _log as _main_log
from HPOBenchExperimentUtils.utils.validation_utils import load_trajectories, \
    load_trajectories_as_df, df_per_optimizer
from HPOBenchExperimentUtils.utils.plotting_utils import color_per_opt, marker_per_opt, plot_dc
from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_setting

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

    plt.figure(figsize=[5*len(stat_dc), 5])

    ax_old = None
    for i, opt in enumerate(stat_dc):
        ax = plt.subplot(1, len(stat_dc), i+1, sharey=ax_old)
        # get queried fidels
        df = stat_dc[opt]
        nseeds = df['id'].unique()
        avg = []
        for seed in nseeds:
            fidels = df[df['id'] == seed]["fidel_values"]
            avg.append(len(fidels))
            steps = df[df['id'] == seed]["total_time_used"]
            label = get_optimizer_setting(opt).get("display_name", opt)

            plt.scatter(steps, fidels, edgecolor=color_per_opt.get(opt, "k"), facecolor="none",
                        marker=marker_per_opt[opt], alpha=0.5,
                        label=label if seed == nseeds[-1] else None)
        plt.xscale("log")
        plt.xlabel("Runtime in seconds")
        plt.legend(title="%g evals on avg" % np.mean(avg))
        ax_old = ax

    plt.ylabel("Fidelity")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{benchmark}_fidel.png')


def plot_overhead(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Start plotting overhead of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")
    benchmark_spec = plot_dc.get(benchmark, {})
    y_best = benchmark_spec.get("ystar_valid", 0)

    plt.figure(figsize=[5, 5])
    a = plt.subplot(111)
    for opt in opt_rh_dc:
        if len(opt_rh_dc[opt]) == 0: continue
        df = load_trajectories(opt_rh_dc[opt])
        df = df_per_optimizer(opt, df, y_best=y_best)
        nseeds = df['id'].unique()
        for seed in nseeds:
            steps = df[df['id'] == seed]["total_time_used"]
            label = get_optimizer_setting(opt).get("display_name", opt)

            benchmark_cost = df[df['id'] == seed]["finish_time"] - df[df['id'] == seed]["start_time"]
            benchmark_cost = np.cumsum(benchmark_cost)
            plt.plot(steps, benchmark_cost, color='k', alpha=0.5, zorder=99,
                     label=benchmark if seed == 0 and opt == list(opt_rh_dc.keys())[0] else None)

            overhead = df[df['id'] == seed]["start_time"] - df[df['id'] == seed]["finish_time"].shift(1)
            overhead = np.cumsum(overhead)
            plt.plot(steps, overhead, color=color_per_opt.get(opt, "k"), linestyle=":", label=label if seed == 0 else None)

            overall_cost = df[df['id'] == seed]["finish_time"] - df[df['id'] == seed]["start_time"].iloc[0]
            overall_cost = np.cumsum(overall_cost)
            plt.plot(steps, overall_cost, color=color_per_opt.get(opt, "k"), alpha=0.5, zorder=99,
                     label="%s overall" % label if seed == 0 else None)
        del df

    a.legend()
    a.grid(which="both", zorder=0)
    a.set_yscale("log")
    a.set_xscale("log")
    a.set_xlabel("Runtime in seconds")
    a.set_ylabel("Cumulated overhead in seconds")
    a.set_xlim([1, a.set_xlim()[1]])
    #a.set_ylim([0.1, 10000])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{benchmark}_overhead.png')


def plot_ecdf(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Start plotting ECDFs for benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")
    benchmark_spec = plot_dc.get(benchmark, {})
    y_best = benchmark_spec.get("ystar_valid", 0)

    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        return xs, ys

    plt.figure(figsize=[5, 5])
    a = plt.subplot(111)

    for opt in opt_rh_dc:
        if len(opt_rh_dc) == 0: continue
        df = load_trajectories(opt_rh_dc[opt])
        df = df_per_optimizer(opt, df, y_best=y_best)

        color = color_per_opt.get(opt, "k")

        obj_vals = df["function_values"]
        x, y = ecdf(obj_vals.to_numpy())
        label = get_optimizer_setting(opt).get("display_name", opt)
        plt.plot(x, y, c=color, linewidth=2, label=label)

        nseeds = df['id'].unique()
        for seed in nseeds:
            obj_vals = df[df['id'] == seed]["function_values"]
            x, y = ecdf(obj_vals.to_numpy())
            plt.plot(x, y, c=color, alpha=0.2)
        del df

    if y_best != 0:
        plt.xlabel("Optimization Regret")
    else:
        plt.xlabel("Optimization objective value")
    plt.ylabel("P(x < X)")
    yscale = benchmark_spec.get("yscale", "log")
    plt.xscale(yscale)
    plt.tight_layout()
    plt.legend()
    plt.grid(b=True, which="both", axis="both", alpha=0.5)
    plt.savefig(Path(output_dir) / f'{benchmark}_ecdf.png')
