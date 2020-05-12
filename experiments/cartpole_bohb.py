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

from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

from hpolib.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from hpolib.util.rng_helper import get_rng
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


def run_experiment(out_path, seed):

    settings = {'min_budget': 1,
                'max_budget': 5,  # Number of Agents, which are trained to solve the cartpole experiment
                'num_iterations': 10,  # Number of HB brackets
                'eta': 3,
                'output_dir': Path(out_path)
                }

    settings.get('output_dir').mkdir(exist_ok=True)

    cs = Benchmark.get_configuration_space()
    seed = get_rng(rng=seed)
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
    parser.add_argument('--out_path', default='./cartpole_smac_hb', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, seed=args.seed)
