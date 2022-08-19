"""
Script to generate ECDF plots for Tabular Benchmarks collected
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Union, List
from matplotlib import pyplot as plt
from hpobench.benchmarks.ml import TabularBenchmark
from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc, unify_layout, benchmark_families

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False


all_task_ids_by_in_mem_size = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212, 168911,
    9981, 168329, 167120, 14965, 146606,  # < 30 MB
    168330, 7592, 9977, 168910, 168335, 146195, 168908, 168331,  # 30-100 MB
    168868, 168909, 189355, 146825, 7593,  # 100-500 MB
    168332, 168337, 168338,  # > 500 MB
    189354, 34539,  # > 2.5k MB
    3945,  # >20k MB
    # 189356  # MemoryError: Unable to allocate 1.50 TiB; array size (256419, 802853) of type float64
]
paper_tasks = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212, 168911,
    9981, 167120, 14965, 146606, 7592, 9977
]
all_task_ids_by_in_mem_size = paper_tasks
ntasks_done = dict(
    svm=29,
    lr=29,
    rf=28,
    xgb=22,
    nn=8
)


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def plot_ecdf_tabular(
        family_name: str, format:str, input_dir: Union[Path, str], output_dir: Union[Path, str]
):
    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    fidelity_name = dict(
        svm="subsample",
        lr="iter",
        rf="n_estimators",
        xgb="n_estimators",
        nn="iter"
    )

    values_per_dataset = dict()
    benchmarks = benchmark_families[family_name]
    for benchmark_name in benchmarks:
        print(benchmark_name)
        model, task_id = benchmark_name.split("_")
        if task not in all_task_ids_by_in_mem_size[:ntasks_done[model]]
            print("Skipping {}...".format(task_id))
            continue
        task_id = int(task_id)
        benchmark_spec = plot_dc.get(benchmark_name, {})
        y_best = benchmark_spec.get("ystar_valid", 0)
        y_max = benchmark_spec.get("y_max", 0)

        benchmark = TabularBenchmark(data_dir=input_dir, model=model, task_id=task_id)
        full_budget = benchmark.table[fidelity_name[model]].max()
        # subsetting for full budget
        table = benchmark.table[benchmark.table[fidelity_name[model]] == full_budget]
        seeds = table.seed.unique()
        values = np.zeros(int(table.shape[0] / len(seeds)))
        # averaging loss across all seeds
        for seed in seeds:
            _table = table[table.seed == seed]
            values += np.array([res["function_value"] for res in _table.result])
        values /= len(seeds)
        values_per_dataset[task_id] = values
        # calculating normalized regrets
        regrets = (values - y_best) / (y_max - y_best)
        # plotting regret ecdf
        x, y = ecdf(regrets)
        plt.plot(x, y, alpha=0.3, color="#984ea3")
    plt.xlabel("Normalized regret", fontsize=25)
    plt.ylabel(r"$P(X \le x)$", fontsize=25)
    plt.savefig(output_dir / "{}.{}".format(family_name, format), bbox_tight="inches")


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default="/work/dlclarge1/mallik-hpobench/opt-results/aug2022/ecdf")
    parser.add_argument('--input_dir', type=str,
                        default="/work/dlclarge1/mallik-hpobench/DataDir/Data/TabularData/")
    parser.add_argument('--model', choices=["svm", "lr", "rf", "xgb", "nn"], default=None)
    parser.add_argument('--format', choices=["png", "pdf"], default="pdf")
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = input_args()

    plot_ecdf_tabular(
        family_name="tabular_{}".format(args.model),
        format=args.format,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
