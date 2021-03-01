import logging
from pathlib import Path
from typing import Union

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.stats as scst

from HPOBenchExperimentUtils import _log as _main_log
from HPOBenchExperimentUtils.utils.validation_utils import load_json_files, \
    load_trajectories_as_df, df_per_optimizer
from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc, color_per_opt, marker_per_opt
from HPOBenchExperimentUtils.utils.runner_utils import get_optimizer_setting, get_benchmark_settings

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)


def plot_fidels(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Plotting evaluated fidelities of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")

    other_stats_dc = dict()
    best_val = 1000
    other_stats_dc["lowest"] = best_val

    plt.figure(figsize=[5*len(opt_rh_dc), 5])
    ax_old = None

    for i, opt in enumerate(opt_rh_dc):
        _log.info("Handling %s" % opt)
        if len(opt_rh_dc) == 0: continue
        other_stats_dc[opt] = defaultdict(list)
        rhs = load_json_files(opt_rh_dc[opt])
        for rh in rhs:
            final_time = rh[-1]["finish_time"] - rh[0]["boot_time"]
            bench_time = rh[-1]["total_time_used"]
            calls = rh[-1]["function_call"]
            other_stats_dc[opt]["final_time"].append(final_time)
            other_stats_dc[opt]["bench_time"].append(bench_time)
            other_stats_dc[opt]["calls"].append(calls)
        df = df_per_optimizer(opt, rhs)
        ax = plt.subplot(1, len(opt_rh_dc), i+1, sharey=ax_old)
        
        thresh = 10000
        if df.shape[0] > thresh:
            sub = df[["fidel_values", "total_time_used"]].sample(n=thresh, random_state=1)
        else:
            sub = df[["fidel_values", "total_time_used"]]
        avg = sub.shape[0] / len(df['id'].unique())
        
        max_f = np.max(sub["fidel_values"])
        vals = np.min(df.query('fidel_values == @max_f')["function_values"])
        best_val = np.min([best_val, vals])
        other_stats_dc["lowest"] = best_val

        label = get_optimizer_setting(opt).get("display_name", opt)
        plt.scatter(sub["total_time_used"], sub["fidel_values"], edgecolor=color_per_opt.get(opt, "k"), facecolor="none",
                    marker=marker_per_opt[opt], alpha=0.5,
                    label=label)
        plt.xscale("log")
        plt.xlabel("Runtime in seconds")
        plt.legend(title="%g evals on avg" % avg)
        ax_old = ax
        del rhs, df, sub

    with open(Path(output_dir) / f'stats_{benchmark}.json', "w") as fh:
        json.dump(other_stats_dc, fh, indent=4, sort_keys=True)

    plt.ylabel("Fidelity")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'fidel_{benchmark}.png')
    plt.close('all')


def plot_overhead(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Start plotting overhead of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")

    plt.figure(figsize=[5, 5])
    a = plt.subplot(111)
    for opt in opt_rh_dc:
        if len(opt_rh_dc) == 0: continue
        rhs = load_json_files(opt_rh_dc[opt])
        df = df_per_optimizer(opt, rhs)
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
            benchmark_cost = np.cumsum(overall_cost)
            plt.plot(steps, overall_cost, color=color_per_opt.get(opt, "k"), alpha=0.5, zorder=99,
                     label="%s overall" % label if seed == 0 else None)

    a.legend()
    a.grid(which="both", zorder=0)
    a.set_yscale("log")
    a.set_xscale("log")
    a.set_xlabel("Runtime in seconds")
    a.set_ylabel("Cumulated overhead in seconds")
    a.set_xlim([1, a.set_xlim()[1]])
    #a.set_ylim([0.1, 10000])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'overhead_{benchmark}.png')
    plt.close('all')


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
        rhs = load_json_files(opt_rh_dc[opt])
        df = df_per_optimizer(opt, rhs, y_best=y_best)
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
    plt.savefig(Path(output_dir) / f'ecdf_{benchmark}.png')


def plot_correlation(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Start plotting corralations for benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")
    benchmark_spec = plot_dc.get(benchmark, {})
    y_best = benchmark_spec.get("ystar_valid", 0)

    conf_dc = defaultdict(dict)
    f_set = []
    for opt in opt_rh_dc:
        if not ("smac" in opt
                or "dehb" in opt
                or "hpbands" in opt):
            _log.info("Skip %s" % opt)
            continue
        _log.info("Read %s" % opt)
        if len(opt_rh_dc[opt]) == 0: continue

        rhs = load_json_files(opt_rh_dc[opt])
        for rh in rhs:
            for record in rh[1:]:
                c = json.dumps(record["configuration"], sort_keys=True)
                f = record['fidelity'][list(record['fidelity'])[0]]
                f_set.append(f)
                conf_dc[c][f] = record["function_value"]

    f_set = np.array(list(set(f_set)))
    f_set.sort()
    # Clean dc:
    to_rm = []
    for c in conf_dc:
        if len(conf_dc[c]) < 2:
            to_rm.append(c)
    for c in to_rm:
        del conf_dc[c]

    # Start with computing correlations
    cors = {}
    for fi, f1 in enumerate(f_set):
        for f2 in f_set[fi+1:]:
            a = []
            b = []
            for c in conf_dc:
                if f1 in conf_dc[c] and f2 in conf_dc[c]:
                    a.append(conf_dc[c][f1])
                    b.append(conf_dc[c][f2])
            c, _ = scst.spearmanr(a, b)
            cors[(f1, f2)] = (c, len(a))

    # Create plot
    plt.figure(figsize=[5, 5])
    a = plt.subplot(111)
    for fi, f in enumerate(f_set):
        plt.scatter(f_set[fi+1:], [cors[(f, f1)][0] for f1 in f_set[fi+1:]], label=f)
    plt.legend()
    plt.xlabel("fidelity")
    plt.ylabel("Spearman correlation coefficient")
    plt.grid(b=True, which="both", axis="both", alpha=0.5)
    plt.savefig(Path(output_dir) / f'correlation_{benchmark}.png')

    # Create table
    df = defaultdict(list)
    for fi, f1 in enumerate(f_set[:-1]):
        for fj, f2 in enumerate(f_set):
            if fj <= fi:
                df[f1].append("-")
            else:
                df[f1].append("%.3g (%d)" % (np.round(cors[f1, f2][0], 3), cors[f1, f2][1]))
    df = pd.DataFrame(df, index=f_set)
    with open(Path(output_dir) / f'correlation_table_{benchmark}.tex', 'w') as fh:
        latex = df.to_latex(index_names=False, index=True)
        fh.write(latex)


def get_stats(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], **kwargs):
    _log.info(f'Start plotting corralations for benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    opt_rh_dc = load_trajectories_as_df(input_dir=input_dir,
                                        which="runhistory")
    benchmark_settings = get_benchmark_settings(benchmark)

    stats = {"lowest_val": 10000000}

    for opt in opt_rh_dc:
        _log.info("Read %s" % opt)
        if len(opt_rh_dc[opt]) == 0: continue
        stats[opt] = {
            "sim_wc_time": [],
            "diff_wc_time": [],
            "n_calls": [],
            "act_wc_time": [],
        }
        for fl in opt_rh_dc[opt]:
            with open(fl, "r") as fh:
                lines = fh.readlines()
            rh = [json.loads(line) for line in lines]

            fids = np.array([list(e["fidelity"].values())[0] for e in rh])
            vals = np.array([list(e["function_value"].values())[0] for e in rh])
            high_fid = max(fids)
            lowest = np.min(vals[fids == high_fid])
            stats["lowest_val"] = np.min(stats["lowest_val"], lowest)

            boot_time = rh[0]["boot_time"]
            sim_wc_time = rh[-1]["total_time_used"]
            diff_wc_time = benchmark_settings["time_limit_in_s"] - sim_wc_time
            n_calls = len(rh)-1
            act_wc_time = rh[-1]["finish_time"] - boot_time
            stats[opt]["sim_wc_time"].append(sim_wc_time)
            stats[opt]["diff_wc_time"].append(diff_wc_time)
            stats[opt]["n_calls"].append(n_calls)
            stats[opt]["act_wc_time"].append(act_wc_time)

    with open(Path(output_dir) / f'stats_{benchmark}.json', 'w') as fh:
        json.dump(stats, fh)
