from typing import List
import numpy as np


def create_trajectory(runhistory: List, bigger_is_better: bool):
    trajectory = []

    inc_value = np.inf
    inc_budget = -np.inf

    for i_record, record in enumerate(runhistory):

        if i_record == 0:
            trajectory.append(record)
            continue

        fidelity = list(record['fidelity'].values())[0]

        if ((bigger_is_better
                and ((abs(fidelity - inc_budget) <= 1e-8 and ((inc_value - record['function_value']) > 1e-8))
                     or (fidelity - inc_budget) > 1e-8))

            or

            (not bigger_is_better
                and ((inc_value - record['function_value']) > 1e-8))):

            inc_value = record['function_value']
            inc_budget = fidelity
            trajectory.append(record)

    return trajectory


