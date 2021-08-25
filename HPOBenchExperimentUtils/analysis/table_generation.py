import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import scipy.stats as scst

from HPOBenchExperimentUtils.utils.validation_utils import load_json_files, \
    load_trajectories_as_df, df_per_optimizer
from HPOBenchExperimentUtils import  _log as _main_log
from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc
from HPOBenchExperimentUtils.utils.runner_utils import get_benchmark_settings

_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)


def write_latex(result_df, output_file, col_list):
    replace_dc = {
        '\\{': "{",
        "\\}": "}",
        "textbf": "\\textbf",
        "underline": "\\underline",
        'xgboostsub': r"\xgboostfrac",
        'xgboostest': r"\xgboostnest",
        'cartpolereduced': r"\cartpole",
        "cartpolefull": "%cartpolefull",
        'BNNOnBostonHousing': r"\bnnboston",
        'BNNOnProteinStructure': r"\bnnprotein",
        'BNNOnYearPrediction': r"\bnnyear",
        'learna': r"\learna",
        'NASCifar10ABenchmark': r"\NASA",
        'NASCifar10BBenchmark': r"\NASB",
        'NASCifar10CBenchmark': r"\NASC",
        'SliceLocalizationBenchmark': r"\slice",
        'ProteinStructureBenchmark': r"\protein",
        'NavalPropulsionBenchmark': r"\naval",
        'ParkinsonsTelemonitoringBenchmark': r"\parkinson",
        'Cifar10NasBench201Benchmark': r"%\nbcifart",
        'Cifar10ValidNasBench201Benchmark': r"\nbcifartv",
        'Cifar100NasBench201Benchmark': r"\nbcifarh",
        'ImageNetNasBench201Benchmark': r"\nbimage",
        "SurrogateSVMBenchmark": r"\nsvmsurro",
        "ParamNetReducedAdultOnTimeBenchmark": r"\paramadult",
        "ParamNetReducedHiggsOnTimeBenchmark": r"\paramhiggs",
        "ParamNetReducedLetterOnTimeBenchmark": r"\paramletter",
        "ParamNetReducedMnistOnTimeBenchmark": r"\parammnist",
        "ParamNetReducedOptdigitsOnTimeBenchmark": r"\paramoptdigits",
        "ParamNetReducedPokerOnTimeBenchmark": r"\parampoker",
        "NASBench1shot1SearchSpace1Benchmark": r"\NASOSOA",
        "NASBench1shot1SearchSpace2Benchmark": r"\NASOSOB",
        "NASBench1shot1SearchSpace3Benchmark": r"\NASOSOC",
    }
    with open(output_file, 'w') as fh:
        latex = result_df.to_latex(index_names=False, index=False, columns=["benchmark"] + col_list)
        for i in replace_dc:
            latex = latex.replace(i, replace_dc[i])
        print(latex)
        fh.write(latex)


def save_median_table(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str], opts: str,
                      unvalidated: bool = True, which: str = "v1",
                      opt_list: Union[List[str], None] = None, thresh=1, **kwargs):

    assert 0 < thresh <= 1, f"thresh needs to be in [0, 1], but is {thresh}"

    _log.info(f'Start creating table of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    unique_optimizer = load_trajectories_as_df(input_dir=input_dir,
                                               which=f'train_{which}' if unvalidated else
                                               f'test_{which}')
    benchmark_spec = plot_dc.get(benchmark, {})
    y_best_val = benchmark_spec.get("ystar_valid", 0)
    y_best_test = benchmark_spec.get("ystar_test", 0)
    benchmark_settings = get_benchmark_settings(benchmark)
    time_limit_in_s = benchmark_settings["time_limit_in_s"]
    cut_time_step = thresh * time_limit_in_s
    _log.info(f"Cut to {thresh} percent -> {time_limit_in_s} sec")

    keys = list(unique_optimizer.keys())
    if opt_list is None:
        opt_list = keys
    result_df = pd.DataFrame()
    for key in opt_list:
        if key not in keys:
            _log.info(f'Skip {key}')
            continue
        trajectories = load_json_files(unique_optimizer[key])
        optimizer_df = df_per_optimizer(
            key=key,
            unvalidated_trajectories=trajectories,
            y_best=y_best_val if unvalidated else y_best_test
        )

        unique_ids = np.unique(optimizer_df['id'])
        for unique_id in unique_ids:
            df = optimizer_df[optimizer_df['id'] == unique_id]
            df2 = pd.DataFrame([[key, unique_id, 0, 0, np.inf, df["fidel_values"].max(), 0, 0, 1], ], columns=df.columns)
            df = df.append(df2, ignore_index=True)
            df = df.sort_values(by='total_time_used')
            df = df.drop(df[df["total_time_used"] > cut_time_step].index)
            last_inc = df.tail(1)
            if len(last_inc) <= 1:
                _log.critical(f"{key} has not enough runs at timestep {cut_time_step}")

            result_df = result_df.append(last_inc)

    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    def lst(x):
        x = np.array(x)
        #x[x < 1e-6] = 1e-6
        return list(x)

    def median(x):
        x = np.array(x)
        #x[x < 1e-6] = 1e-6
        return np.median(x)

    # q1 = lambda x: x.quantile(0.25)
    # q3 = lambda x: x.quantile(0.75)
    aggregate_funcs = [median, q1, q3, lst]
    result_df = result_df.groupby('optimizer').agg({'function_values': aggregate_funcs,
                                                    'total_time_used': ['median']})
    result_df.columns = ["_".join(x) for x in result_df.columns.ravel()]

    # Compute some statistics
    opt_keys = list(result_df.index)
    opt_keys.sort()

    # get best optimizer
    best_opt = opt_keys[result_df["function_values_median"].argmin()]
    best_opt_ls = [best_opt, ]
    best_val = np.array(result_df["function_values_lst"][best_opt])
    _log.info(f"{best_opt} is the best optimizer; found {len(best_val)} runs")

    not_worse = []
    for opt in opt_keys:
        if opt == best_opt: continue
        opt_val = np.array(result_df["function_values_lst"][opt])
        if not len(opt_val) == len(best_val):
            _log.warning(f"There are not {len(best_val)} but {len(opt_val)} repetitions for {opt}")
            continue

        if np.sum(best_val - opt_val) == 0:
            # Results are identical
            best_opt_ls.append(opt)
        else:
            # The two-sided test has the null hypothesis that the median of the differences is zero
            # against the alternative that it is different from zero.
            s, p = scst.wilcoxon(best_val, opt_val, alternative="two-sided")
            if p > 0.05:
                not_worse.append(opt)

    for opt in opt_keys:
        val = result_df["function_values_median"][opt]

        if val < 1e-3:
            val = "%.2e" % val
        else:
            val = "%.3g" % np.round(val, 3)

        if opt in best_opt_ls:
            val = r"underline{textbf{%s}}" % val
        elif opt in not_worse:
            val = r"underline{%s}" % val

        result_df.loc[opt, "value"] = val

    # result_df = result_df.round({
    #    "function_values_median": 3,
    #    "function_values_q1": 2,
    #    "function_values_q3": 2,
    #    "total_time_used_median": 0,
    # })

    # select final cols
    result_df['optimizer'] = result_df.index
    result_df = result_df[["value"]]
    result_df = result_df.transpose()
    result_df["benchmark"] = benchmark

    header = ["benchmark"] + opt_keys
    result_df = result_df[header]

    val_str = 'unvalidated' if unvalidated else 'validated'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if thresh < 1:
        output_file = Path(output_dir) / f'result_table_{benchmark}_{val_str}_{which}_{int(thresh*100)}_{opts}.tex'
    else:
        output_file = Path(output_dir) / f'result_table_{benchmark}_{val_str}_{which}_{opts}.tex'

    not_available = [opt for opt in opt_list if opt not in result_df.columns]
    for col in not_available:
        result_df[col] = -1
    write_latex(result_df=result_df, output_file=output_file, col_list=opt_list)
