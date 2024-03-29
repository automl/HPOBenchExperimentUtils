import logging
from pathlib import Path
from typing import Union, Dict
import os
import numpy as np
import json

from HPOBenchExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOBenchExperimentUtils.utils.dragonfly_utils import \
    configspace_to_dragonfly, load_dragonfly_options, generate_trajectory, change_cwd
from dragonfly import minimise_function, \
    minimise_multifidelity_function

from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

from ConfigSpace import Configuration

_log = logging.getLogger(__name__)

class DragonflyOptimizer(Optimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

        fidel_space = self.benchmark.get_fidelity_space()
        config, domain_parsers, fidelity_parsers, fidelity_costs = \
            configspace_to_dragonfly(domain_cs=self.cs, fidelity_cs=fidel_space)

        _log.debug("Based on the HPOBench Benchmark, generated the config:\n%s" % json.dumps(config, indent=4))

        self.options, self.config = load_dragonfly_options(hpoexp_settings=self.settings, config=config)

        _log.debug("Dragonfly optimizer options are:\n%s" % json.dumps(self.options.__dict__, indent=4))
        _log.debug("Dragonfly optimizer config is:\n%s" % str(self.config.__dict__))

        if self.options.max_capital < 0:
            raise ValueError('max_capital (time or number of evaluations) must be positive.')

        self.is_mf = fidelity_parsers is not None

        def parse_domain(x):
            return Configuration(
                configuration_space=self.cs,
                values={parser[0]: parser[1](val) for parser, val in zip(domain_parsers, x)}
            )

        def objective(x):
            conf = parse_domain(x)
            _log.debug("Calling no-fidelity objective with configuration %s." % str(conf.get_dictionary()))
            run_id = Optimizer._id_generator()
            return self.benchmark.objective_function(configuration=conf, configuration_id=run_id)['function_value']

        self.parse_domain = parse_domain
        self.objective = objective

        if self.is_mf:
            def parse_fidelities(z):
                ret = {parser[0]: parser[1](val) for parser, val in zip(fidelity_parsers, z)}
                return ret

            def objective_mf(z, x):
                run_id = Optimizer._id_generator()
                conf = parse_domain(x)
                fidels = parse_fidelities(z)
                _log.debug("Calling multi-fidelity objective with configuration %s at fidelity %s" % (
                    conf.get_dictionary(), fidels))
                ret = self.benchmark.objective_function(configuration=conf,
                                                        configuration_id=run_id,
                                                        fidelity=fidels,
                                                        **self.settings_for_sending,
                                                        )
                _log.debug("multi-fidelity objective returned %s" % (str(ret)))
                return ret['function_value']

            def cost(z):
                ret = sum([c(v) for c, v in zip(fidelity_costs, z)]) / float(len(fidelity_costs)) + 1e-6
                return ret

            self.parse_fidelities = parse_fidelities
            self.objective_mf = objective_mf
            self.cost = cost

    def setup(self):
        pass

    def run(self) -> Path:

        # Change the current working directory to a unique temporary location in order to avoid any
        # huge messes due to dragonfly's multi-process communication system

        old_cwd = Path().cwd()
        change_cwd()

        np.random.seed(self.rng)

        if self.is_mf:
            _log.info('Minimising multi-fidelity function on\n Fidelity-Space: %s.\n Domain: %s.' % (
                self.config.fidel_space, self.config.domain))
            opt_val, opt_pt, history = minimise_multifidelity_function(
                self.objective_mf, fidel_space=None, domain=None,
                fidel_to_opt=self.config.fidel_to_opt, fidel_cost_func=self.cost,
                max_capital=self.options.max_capital, capital_type=self.options.capital_type,
                opt_method=self.options.opt_method, config=self.config, options=self.options,
                reporter=self.options.report_progress, worker_manager="synthetic")
        else:
            # test_obj = objective(test_point)
            _log.info('Minimising function on Domain: %s.' % self.config.domain)
            opt_val, opt_pt, history = minimise_function(
                self.objective, domain=None, max_capital=self.options.max_capital,
                capital_type=self.options.capital_type, opt_method=self.options.opt_method,
                config=self.config, options=self.options, reporter=self.options.report_progress,
                worker_manager="synthetic")

        # Go back to the original working directory we started from
        os.chdir(old_cwd)

        generate_trajectory(history, save_file=self.output_dir / f"dragonfly_traj.json",
                            is_cp=True if isinstance(history.query_qinfos[0].point, list) else False,
                            history_file=self.output_dir / "dragonfly_history.json")

        # return self.optimizer_settings['output_dir']
