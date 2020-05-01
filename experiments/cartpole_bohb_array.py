"""
SMAC on Cartpole with BOHB
==========================

This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.

Please install the necessary dependencies via ``pip install .[cartpole_example]``
and the HPOlib3 ``pip install <dir of hpolib>``
"""
import logging
import pickle

from collections import OrderedDict
from pathlib import Path
from trajectory_parser import BOHBReader

logging.basicConfig(level=logging.DEBUG)

from hpolib.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from hpolib.util.rng_helper import get_rng
from hpolib.util.example_utils import set_env_variables
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker

logger = logging.getLogger('BOHB on cartpole')
# set_env_variables()


class CustomWorker(Worker):
    def __init__(self, seed, max_budget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.max_budget = max_budget

    def compute(self, config, budget, **kwargs):
        b = Benchmark(rng=self.seed, max_budget=self.max_budget)
        result_dict = b.objective_function(config, budget=int(budget))
        return {'loss': result_dict['function_value'],
                'info': {'cost': result_dict['cost'],
                         'budget': result_dict['budget']}}


def run_experiment(out_path):

    settings = {'min_budget': 1,
                'max_budget': 5,  # Number of Agents, which are trained to solve the cartpole experiment
                'num_iterations': 10,  # Number of HB brackets
                'eta': 3,
                'output_dir': Path(out_path)
                }

    settings.get('output_dir').mkdir(exist_ok=True)

    cs = Benchmark.get_configuration_space()
    seed = get_rng(rng=0)
    run_id = 'BOHB_on_cartpole'

    result_logger = hpres.json_result_logger(directory=str(settings.get('output_dir')), overwrite=True)

    ns = hpns.NameServer(run_id=run_id, host='localhost', working_directory=str(settings.get('output_dir')))
    ns_host, ns_port = ns.start()

    worker = CustomWorker(seed=seed,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          run_id=run_id,
                          max_budget=settings.get('max_budget'))
    worker.run(background=True)

    master = BOHB(configspace=cs,
                  run_id=run_id,
                  host=ns_host,
                  nameserver=ns_host,
                  nameserver_port=ns_port,
                  eta=settings.get('eta'),
                  min_budget=settings.get('min_budget'),
                  max_budget=settings.get('max_budget'),
                  result_logger=result_logger)

    result = master.run(n_iterations=settings.get('num_iterations'))
    master.shutdown(shutdown_workers=True)
    ns.shutdown()

    with open(settings.get('output_dir') / 'results.pkl', 'wb') as f:
        pickle.dump(result, f)

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    inc_value = result.get_runs_by_id(incumbent)[-1]['loss']
    inc_cfg = id2config[incumbent]['config']

    logger.info(f'Inc Config:\n{inc_cfg}\n'
                f'with Performance: {inc_value:.2f}')

    benchmark = Benchmark()
    incumbent_result = benchmark.objective_function(config=inc_cfg)
    print(incumbent_result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib - BOHB',
                                     description='HPOlib3 with BOHB on Cartpole',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./cartpole_bohb/', type=str)
    parser.add_argument('--array_id', default=0, type=int)
    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--validate', dest='validate', action='store_true')
    args = parser.parse_args()

    out_path = Path(args.out_path) / f'run-{args.array_id}'
    out_path.mkdir(exist_ok=True, parents=True)
    if args.optimize:
        run_experiment(out_path=out_path)

    traj_reader = BOHBReader()
    traj_reader.read(out_path)
    traj_reader.get_trajectory()
    traj_reader.export_trajectory(output_path=out_path / 'unified_traj.json')

    if args.validate:
        traj_ids = traj_reader.get_configuration_ids_trajectory()
        traj_configs_to_validate = OrderedDict({traj_id: traj_reader.config_ids_to_configs[traj_id][0]
                                                for traj_id in traj_ids})
        traj_validated_loss = OrderedDict({str(traj_id): -1234 for traj_id in traj_ids})

        benchmark = Benchmark()
        # for traj_id in traj_configs_to_validate:
        #     result = benchmark.objective_function(configuration=traj_configs_to_validate[traj_id], budget=1)
        #     traj_validated_loss[traj_id] = result['function_value']
        #     traj_reader.add_validated_trajectory(traj_validated_loss)

        # This one is if you need to save the validated runs.
        #     traj_validated_loss[str(traj_id)] = result['function_value']

        # with (out_path / 'validated_traj.json').open('w') as fh:
        #     import json_tricks as json
        #     json.dump(traj_validated_loss, fh)

        with (out_path / 'validated_traj.json').open('r') as fh:
            import json_tricks as json
            traj_validated_loss = json.load(fh)

        traj_validated_loss_ = OrderedDict()
        for key in traj_validated_loss.keys():
            k = key.replace(' ', '').replace('(', '').replace(')', '')
            k = k.split(',')
            k = tuple(int(t) for t in k)
            traj_validated_loss_[k] = traj_validated_loss[key]
        traj_validated_loss = traj_validated_loss_

        traj_reader.add_validated_trajectory(traj_validated_loss)

        df_validated = traj_reader.get_validated_trajectory_as_dataframe()
        df = traj_reader.get_trajectory_as_dataframe()

        pass



