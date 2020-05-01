import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Dict, Optional
from trajectory_parser.run_result import Run
from pprint import pprint

class ResultReader:
    def __init__(self):
        self.config_ids_to_configs = {}
        self.results = []
        self.trajectory = []

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
        trajectory = self.get_trajectory()
        trajectory = np.array([[finish_time, run.loss, run.budget if meaningful_budget else 1, run.config_id]
                               for finish_time, run in trajectory])
        trajectory = pd.DataFrame(trajectory, columns=['wallclock_time', 'cost', 'budget', 'config_id'])
        trajectory = trajectory.set_index('wallclock_time')
        trajectory = trajectory.add_suffix(suffix)
        return trajectory

    def export_trajectory(self, output_path):
        assert len(self.trajectory) != 0, "No trajectory available. Please read-in trajectory before."

        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / 'trajparser_traj.json'

        # add incumbent to info
        lines = []
        for wallclock_time, entry in self.trajectory:
            line = {"wallclock_time": entry.relative_finish_time,
                    "evaluations": entry.info.get('evaluations', -1),
                    "cost": entry.loss,
                    "incumbent": self.config_ids_to_configs[entry.config_id],
                    "origin": entry.info.get('origin')}
            lines.append(json.dumps(line) + '\r\n')

        with output_path.open('w', encoding='utf-8') as fh:
            fh.writelines(lines)

    def __repr__(self):
        return f'Reader: {len(self.results)} results from {len(self.config_ids_to_configs)}\n' \
               f'Trajectory: {self.trajectory}]'


class SMACReader(ResultReader):
    def __init__(self):
        super(SMACReader, self).__init__()

    def read(self, file_path: Union[str, Path]):
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

        config_ids_to_configs = {}
        results = []
        with file_path.open('r') as fh:
            for i, line in enumerate(fh.readlines()):
                run_dict = json.loads(line)
                config_ids_to_configs[i] = run_dict.get('incumbent')
                run = Run()
                run.set_values_smac(config_id=i,
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

    def get_trajectory_as_dataframe(self, meaningful_budget: bool = False, suffix: Optional[str] = '_smac') \
            -> pd.DataFrame:
        return super(SMACReader, self).get_trajectory_as_dataframe(meaningful_budget=meaningful_budget, suffix=suffix)


class BOHBReader(ResultReader):
    def __init__(self):
        super(BOHBReader, self).__init__()

    def read(self, file_path: Union[str, Path]):
        file_path = Path(file_path)
        self.config_ids_to_configs = self._read_bohb_confs(file_path)
        self.results = self._read_bohb_res(file_path)

    def get_trajectory_as_dataframe(self, meaningful_budget: bool = True, suffix: Optional[str] = '_bohb') \
            -> pd.DataFrame:
        return super(BOHBReader, self).get_trajectory_as_dataframe(meaningful_budget=meaningful_budget, suffix=suffix)

    def _read_bohb_confs(self, file_path: Path) -> Dict:
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
