import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from trajectory_parser import SMACReader, BOHBReader


file_path = Path('./example_data/cartpole_smac_hb/run_1608637542/')
reader = SMACReader()
reader.read(file_path)
smac_traj = reader.get_trajectory_as_dataframe()


file_path = Path('./example_data/cartpole_bohb/')
reader = BOHBReader()
reader.read(file_path)
bohb_traj = reader.get_trajectory_as_dataframe()

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
