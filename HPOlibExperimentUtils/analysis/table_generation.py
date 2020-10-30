import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from HPOlibExperimentUtils.utils.validation_utils import load_trajectories, \
    load_trajectories_as_df, df_per_optimizer
from HPOlibExperimentUtils import _default_log_format, _log as _main_log


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
