import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as scst

from HPOBenchExperimentUtils.utils.validation_utils import load_trajectories, \
    load_trajectories_as_df, df_per_optimizer
from HPOBenchExperimentUtils import _default_log_format, _log as _main_log
from HPOBenchExperimentUtils.utils.plotting_utils import plot_dc


_main_log.setLevel(logging.DEBUG)
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


def save_table(benchmark: str, output_dir: Union[Path, str], input_dir: Union[Path, str],
               unvalidated: bool = True, **kwargs):
    _log.info(f'Start creating table of benchmark {benchmark}')
    input_dir = Path(input_dir) / benchmark
    assert input_dir.is_dir(), f'Result folder doesn\"t exist: {input_dir}'
    unique_optimizer = load_trajectories_as_df(input_dir=input_dir,
                                               which="train" if unvalidated else "test")
    benchmark_spec = plot_dc.get(benchmark, {})
    y_best_val = benchmark_spec.get("ystar_valid", 0)
    y_best_test = benchmark_spec.get("ystar_test", 0)

    keys = list(unique_optimizer.keys())
    result_df = pd.DataFrame()
    for key in keys:
        trajectories = load_trajectories(unique_optimizer[key])
        optimizer_df = df_per_optimizer(
            key=key,
            unvalidated_trajectories=trajectories,
            y_best=y_best_val if unvalidated else y_best_test
        )

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

    def lst(x):
        return list(x)

    # q1 = lambda x: x.quantile(0.25)
    # q3 = lambda x: x.quantile(0.75)
    aggregate_funcs = ['median', q1, q3, lst]
    result_df = result_df.groupby('optimizer').agg({'function_values': aggregate_funcs,
                                                    'total_time_used': ['median']})
    result_df.columns = ["_".join(x) for x in result_df.columns.ravel()]

    # Compute some statistics
    opt_keys = list(result_df.index)
    opt_keys.sort()
    # get best optimizer
    best_opt = opt_keys[result_df["function_values_median"].argmin()]
    best_val = result_df["function_values_lst"][best_opt]
    print("%s is the best optimizer" % best_opt)

    not_worse = [best_opt, ]
    for opt in opt_keys:
        if opt == best_opt: continue
        opt_val = result_df["function_values_lst"][opt]
        assert len(opt_val) == len(best_val) == 24
        # The two-sided test has the null hypothesis that the median of the differences is zero
        # against the alternative that it is different from zero.
        s, p = scst.wilcoxon(best_val, opt_val, alternative="two-sided")
        if p < 0.05:
            not_worse.append(opt)

    result_df = result_df.round({
        "function_values_median": 2,
        "function_values_q1": 2,
        "function_values_q3": 2,
        "total_time_used_median": 0,
    })

    result_df.loc[best_opt, "function_values_median"] = r"textbf{%s}" % result_df["function_values_median"][best_opt]
    for opt in not_worse:
        result_df.loc[opt, "function_values_median"] = r"underline{%s}" % result_df["function_values_median"][opt]

    result_df["value"] = result_df['function_values_median'].astype(str) + " [" + result_df['function_values_q1'].astype(str) + ", " + \
                             result_df['function_values_q3'].astype(str) + "]"

    # select final cols
    result_df['optimizer'] = result_df.index
    result_df = result_df[["value"]]
    result_df = result_df.transpose()
    result_df["benchmark"] = benchmark

    header = ["benchmark"] + opt_keys
    result_df = result_df[header]

    val_str = 'unvalidated' if unvalidated else 'validated'
    with open(Path(output_dir) / f'{benchmark}_{val_str}_result_table.tex', 'w') as fh:
        latex = result_df.to_latex(index_names=False, index=False)
        latex = latex.replace('\\{', "{").replace("\\}", "}").replace("underline", "\\underline").replace("textbf", "\\textbf")
        fh.write(latex)
