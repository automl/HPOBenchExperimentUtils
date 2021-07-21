from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)



def create_trajectory(runhistory: List, bigger_is_better: bool, main_fidelity: str = None):

    trajectory = []

    inc_value = np.inf
    inc_budget = -np.inf

    for i_record, record in enumerate(runhistory):

        if i_record == 0:
            trajectory.append(record)
            continue

        if len(record['fidelity']) == 0:
            inc_budget = fidelity = 0
        elif len(record['fidelity']) > 1 and main_fidelity is None:
            raise ValueError('We found multiple fidelities, but no main fidelity is specified.')
        else:
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


