import json
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd

from HPOlibExperimentUtils.core.run_result import Run


class ResultReader:
    def __init__(self):
        self.config_ids_to_configs = {}
        self.results = []
        self.trajectory = []
        self.validated_trajectory = []

    def read(self, file_path: Union[str, Path]):
        raise NotImplementedError

    def get_trajectory(self) -> List:
        assert len(self.results) != 0, "No results available. Please call ResultReader.read() before."

        trajectory = []
        for i, run in enumerate(self.results):
            if len(trajectory) == 0:
                trajectory.append((run.get_relative_finish_time(), run))
                continue

            if trajectory[-1][1].budget < run.budget:
                trajectory.append((run.get_relative_finish_time(), run))
                continue

            if trajectory[-1][1].budget == run.budget and trajectory[-1][1].loss > run.loss:
                trajectory.append((run.get_relative_finish_time(), run))

        self.trajectory = trajectory
        return trajectory

    def get_trajectory_as_dataframe(self, meaningful_budget: bool = True, suffix: Optional[str] = '') -> pd.DataFrame:
        return self._get_trajectory_as_dataframe(meaningful_budget, suffix, validated=False)

    def get_validated_trajectory_as_dataframe(self, meaningful_budget: bool = True, suffix: Optional[str] = '') \
            -> pd.DataFrame:
        return self._get_trajectory_as_dataframe(meaningful_budget, suffix, validated=True)

    def _get_trajectory_as_dataframe(self, meaningful_budget: bool = True, suffix: Optional[str] = '',
                                     validated: bool = False) -> pd.DataFrame:
        trajectory = self.get_trajectory() if not validated else self.validated_trajectory
        assert len(trajectory) != 0, 'Please first read in run data before extracting the trajectory'

        trajectory = np.array([[finish_time, run.loss, run.budget if meaningful_budget else 1, run.config_id]
                               for finish_time, run in trajectory])
        trajectory = pd.DataFrame(trajectory, columns=['wallclock_time', 'cost', 'budget', 'config_id'])
        trajectory = trajectory.set_index('wallclock_time')
        trajectory = trajectory.add_suffix(suffix)
        return trajectory

    def export_trajectory(self, output_path: Path):
        self._export_trajectory(output_path, validated=False)

    def export_validated_trajectory(self, output_path: Path):
        self._export_trajectory(output_path, validated=True)

    def _export_trajectory(self, output_path: Path, validated: Union[bool, None] = False):
        trajectory = self.trajectory if not validated else self.validated_trajectory

        assert len(trajectory) != 0, "No trajectory available. Please read-in trajectory before."

        output_path = Path(output_path)
        if output_path.is_dir():
            val_str = 'validated_' if validated else ''
            output_path = output_path / f'traj_{val_str}hpolib.json'

        # add incumbent to info
        lines = []
        for wallclock_time, entry in trajectory:
            line = {"wallclock_time": entry.relative_finish_time,
                    "evaluations": entry.info.get('evaluations', -1),
                    "cost": entry.loss,
                    "incumbent": self.config_ids_to_configs[entry.config_id],
                    "origin": entry.info.get('origin')}
            lines.append(json.dumps(line) + '\r\n')

        with output_path.open('w', encoding='utf-8') as fh:
            fh.writelines(lines)

    def get_configuration_ids_trajectory(self) -> List:
        assert len(self.trajectory) != 0, 'Trajectory is still empty! Read Trajectory before calling this function'
        return [entry.config_id for _, entry in self.trajectory]

    def get_configurations_trajectory(self) -> List:
        assert len(self.trajectory) != 0, 'Trajectory is still empty! Read Trajectory before calling this function'
        return [self.config_ids_to_configs[config_id] for config_id in self.get_configuration_ids_trajectory()]

    def add_validated_trajectory(self, validated_values: Dict):
        self.validated_trajectory = deepcopy(self.trajectory)
        for i, (_, entry) in enumerate(self.validated_trajectory):
            if i == 0:
                continue
            entry.loss = validated_values[entry.config_id]

    def __repr__(self) -> str:
        return f'Reader: {len(self.results)} results from {len(self.config_ids_to_configs)}\n' \
               f'Trajectory: {self.trajectory}]'


class SMACReader(ResultReader):
    def __init__(self):
        super(SMACReader, self).__init__()

    def read(self, file_path: Union[str, Path]):
        """
        Reads in the trajecotry file in SMACs Trajectory format and stores the trajectory in the SMACReader Object.
        This function does not automatically extract also the trajectory. A separate call is necessary.

        Example for SMAC trajectory file format  in examples / examples_data/cartpole_smac_hb/run_1608637542/traj.json

        TODO: Write a read method for the validated trajectory.

        Parameters
        ----------
        file_path : Path
            Either:
            a path to the directory, where a 'traj.json' file or 'traj_aclib2.json' file is. It first tries to
            parse 'traj.json' then 'traj_aclib2.json'. If no such file was found, an exception will be raised.

            Or:
            a path to the specific trajectory file, e.g. /paht/to/own_trajectory_file.json. The file has also to be in
            SMAC trajectory file format.
        """
        file_path = Path(file_path)

        if file_path.is_dir():
            traj_cands = list(file_path.glob('*traj*.json'))
            assert len(traj_cands) != 0, f"no trajectory file in {file_path} found. Please give a direct path to the " \
                                         f"json-trajectory file"

            traj_cands_names = [p.name for p in traj_cands]
            done = False
            try:
                file_path = traj_cands[traj_cands_names.index('traj.json')]
                done = True
            except ValueError:
                pass
            if not done:
                try:
                    file_path = traj_cands[traj_cands_names.index('traj_aclib2.json')]
                    done = True
                except ValueError:
                    pass
            if not done:
                file_path = traj_cands[0]

        config_id = 0
        config_ids_to_configs = {}
        configs_to_config_ids = {}
        results = []
        with file_path.open('r') as fh:
            for i, line in enumerate(fh.readlines()):
                run_dict = json.loads(line)
                if isinstance(run_dict['incumbent'], list):
                    run_dict['incumbent'] = run_dict['incumbent'][0]

                if run_dict.get('incumbent') not in config_ids_to_configs.values():
                    config_ids_to_configs[config_id] = run_dict.get('incumbent')
                    configs_to_config_ids[str(run_dict.get('incumbent'))] = config_id
                    config_id += 1

                run = Run()
                run.set_values_smac(config_id=configs_to_config_ids[str(run_dict.get('incumbent'))],
                                    budget=i,
                                    wallclock_time=run_dict.get('wallclock_time'),
                                    cost=run_dict.get('cost'),
                                    info={'origin': run_dict.get('origin'),
                                          'evaluations': run_dict.get('evaluations')
                                          }
                                    )
                results.append(run)

        self.config_ids_to_configs = config_ids_to_configs
        self.results = results

    def get_trajectory_as_dataframe(self, meaningful_budget: Union[bool, None] = False,
                                    suffix: Union[str, None] = '_smac') -> pd.DataFrame:
        """
        Returns the trajectory as Dataframe. It is important to call reader.read() before!

        Parameters
        ----------
        meaningful_budget : Union[bool, None]
            This parameter specifies if the budget is an acutal value or not. In SMAC's trajectory no budget is
            available. In this case, we fill the budget with the position of the id in ascending order.

             However, the budget in BOHB is the actual budget from the optimization task.
        suffix : Union[str, None]
            The suffix which is appended to each column name in the resulting data frame.

        Returns
        -------
        pd.DataFrame
        """
        return super(SMACReader, self).get_trajectory_as_dataframe(meaningful_budget=meaningful_budget, suffix=suffix)

    def get_validated_trajectory_as_dataframe(self, meaningful_budget: Union[bool, None] = False,
                                              suffix: Union[str, None] = '_smac_valid') -> pd.DataFrame:
        """
        Returns the validated trajectory as Dataframe. It is important to call reader.read() before!

        Parameters
        ----------
        meaningful_budget : Union[bool, None]
            This parameter specifies if the budget is an acutal value or not. In SMAC's trajectory no budget is
            available. In this case, we fill the budget with the position of the id in ascending order.

             However, the budget in BOHB is the actual budget from the optimization task.
        suffix : Union[str, None]
            The suffix which is appended to each column name in the resulting data frame.

        Returns
        -------
        pd.DataFrame
        """
        return super(SMACReader, self).get_validated_trajectory_as_dataframe(meaningful_budget=meaningful_budget,
                                                                             suffix=suffix)


class BOHBReader(ResultReader):
    def __init__(self):
        super(BOHBReader, self).__init__()

    def read(self, file_path: Union[str, Path]):
        """
        Reads in the  BOHB trajecotry file and parses it to SMAC's Trajectory format. The trajectory
        will be stored in the Reader Object.
        This function does not automatically extract also the trajectory. A separate call is necessary.

        Parameters
        ----------
        file_path : Path
            Path to the directory in which the Bohb run results are stored (configs.json, results. json).
        """
        file_path = Path(file_path)
        self.config_ids_to_configs = self._read_bohb_confs(file_path)
        self.results = self._read_bohb_res(file_path)

    def get_trajectory_as_dataframe(self, meaningful_budget: Union[bool, None] = True,
                                    suffix: Optional[str] = '_bohb') -> pd.DataFrame:
        """
        Returns the trajectory as Dataframe. It is important to call reader.read() before!

        Parameters
        ----------
        meaningful_budget : Union[bool, None]
            Since the budgets in BOHBS result are meaningful. This option should be always true!
        suffix : Union[str, None]
            The suffix which is appended to each column name in the resulting data frame.

        Returns
        -------
        pd.DataFrame
        """
        return super(BOHBReader, self).get_trajectory_as_dataframe(meaningful_budget=meaningful_budget, suffix=suffix)

    def get_validated_trajectory_as_dataframe(self, meaningful_budget: Union[bool, None] = True,
                                              suffix: Optional[str] = '_bohb_valid') -> pd.DataFrame:
        """
        Returns the validated trajectory as Dataframe.
        It is important to call reader.read() before on the validated trajectory.

        Parameters
        ----------
        meaningful_budget : Union[bool, None]
            Since the budgets in BOHBS result are meaningful. This option should be always true!
        suffix : Union[str, None]
            The suffix which is appended to each column name in the resulting data frame.

        Returns
        -------
        pd.DataFrame
        """
        return super(BOHBReader, self).get_validated_trajectory_as_dataframe(meaningful_budget=meaningful_budget,
                                                                             suffix=suffix)

    def _read_bohb_confs(self, file_path: Path) -> Dict:
        """ Helper function to read in the bohb configurations from the configs.json file """
        config_ids_to_configs = {}
        with (file_path / 'configs.json').open('r') as fh:
            for i, line in enumerate(fh.readlines()):
                line = json.loads(line)

                if len(line) == 2:
                    (config_id, config), config_info = line, 'N/A'
                if len(line) == 3:
                    config_id, config, config_info = line

                config_ids_to_configs[tuple(config_id)] = [config, config_info]
                if i == 0:
                    config_ids_to_configs[(-1, -1, -1)] = [config, config_info]

        return config_ids_to_configs

    def _read_bohb_res(self, file_path: Path) -> List:
        """ Helper function to read in the bohb results from the results.json file """

        results = []
        start_time = 0
        with (file_path / 'results.json').open('r') as fh:
            # SMAC starts with an incumbent with cost infinity --> Append a starting run.
            run = Run()
            run.set_values_bohb(config_id=[-1, -1, -1], budget=0, time_stamps={'finished': 0},
                                result={'loss': 2.147484e+09}, exception="", global_start_time=0, info={})
            results.append(run)

            for i, line in enumerate(fh.readlines()):
                config_id, budget, time_stamps, result, exception = json.loads(line)
                start_time = time_stamps.get('started') if i == 0 else start_time
                _model_based_pick = self.config_ids_to_configs.get(tuple(config_id))[1].get('model_based_pick')
                origin = 'Model' if _model_based_pick else 'Random'

                run = Run()
                run.set_values_bohb(config_id, budget, time_stamps, result, exception, global_start_time=start_time,
                                    info={'exception': exception,
                                          'origin': origin,
                                          })
                results.append(run)
        return results
