from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_trajectory(runhistory: List, bigger_is_better: bool, main_fidelity: Optional[str] = None):
    """
    Given the runhistory, create the trajecotry of the incumbent (best) configurations.
    We are using often use multi-fidelity benchmarks. In this case, we evaluate a configuration
    for increasing fidelities. Because reduced (not the full) fidelities are only a estimation of the final performance
    of a configuration, there are two possible ways to create a trajectory here.

    1) Bigger_is_better = True: (Treat lower fidelities as estimations)
    We mark a new configuration as incumbent if its performance is better than the current incumbent's performance on it
    highest so far seen fidelity.

    2) Bigger_is_better = False: (Ignore the fidelities)
    Go through the runhistory and set the incumbent to the configuration with the best performance. Ignore in this case
    the fidelities.

    Parameters
    ----------
    runhistory : List[Dict]
    bigger_is_better: bool
        See explanation above.
    main_fidelity : str
        Name of the fidelity that was used for estimating the final performance. Some benchmarks have mulitple
        fidelities. Currently, we always need a "main fidelitiy".
        Raise an error if there are multiple fidelities in the run_history but no main_fidelity is given.

    Returns
    -------

    """
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


