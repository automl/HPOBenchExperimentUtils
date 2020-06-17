"""
This package can be used to read in the run results from SMAC or BOHB and transform it to a equal looking trajectory.
In this example, it is shown how to instantiate the different Reader and export the trajectories.

Normally, those functions are not necessary, since the 'run_experiment.py' and 'validate_experiment.py' functions
perform this automatically.
"""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from HPOlibExperimentUtils import SMACReader, BOHBReader

# Read in the SMAC trajectory
file_path = Path('./example_data/cartpole_smac_hb/run_1608637542/')
smac_reader = SMACReader()
smac_reader.read(file_path)

# Transform it to a dataframe.
smac_traj = smac_reader.get_trajectory_as_dataframe()

# Do the same for the BOHB results. The file path points to the directory, which contains BOHB's config.json and
# result.json
file_path = Path('./example_data/cartpole_bohb/')
bohb_reader = BOHBReader()
bohb_reader.read(file_path)
bohb_traj = bohb_reader.get_trajectory_as_dataframe()

# This function writes the extracted trajectories to file
smac_reader.export_trajectory(Path('./hpolib_traj_smac.json'))
bohb_reader.export_trajectory(Path('./hpolib_traj_bohb.json'))

# For demonstration purpose, we plot the different trajectories.
df = pd.concat([smac_traj, bohb_traj], axis=1, sort=True)
df = df.ffill().bfill()

fig, ax = plt.subplots()
x = df.index.values

for index in ['cost_smac', 'cost_bohb']:
    ax.step(x, df[index].values, label=index)

ax.set_ylabel('Average Runlength')
ax.set_xlabel('Wallclock Time')
ax.set_ylim([0, 4000])
ax.legend()
plt.show()
plt.close()
