import logging
from pathlib import Path
from typing import Union, Dict
import os

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOlibExperimentUtils.utils.dragonfly_utils import \
    configspace_to_dragonfly, load_dragonfly_options, generate_trajectory, change_cwd
from dragonfly import minimise_function, \
    minimise_multifidelity_function

from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient

from ConfigSpace import Configuration

logger = logging.getLogger('Optimizer')


class DragonflyOptimizer(Optimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)

    def setup(self):
        pass

    def run(self) -> Path:

        # TODO: Update to include constraints
        # TODO: Include usage of RNG for consistency
        # TODO: Check for benchmarks that don't have a fidelity space
        # TODO: Read dragonfly optimizer settings
        fidel_space = self.benchmark.get_fidelity_space()
        config, domain_parsers, fidelity_parsers, fidelity_costs = \
            configspace_to_dragonfly(domain_cs=self.cs, fidely_cs=fidel_space)

        try:
            budget = self.settings.get("time_limit_in_s")
            init_frac = self.settings.get("init_capital_frac")
        except KeyError as e:
            raise RuntimeError("Could not read a time limit for the optimizer.") from e

        # TODO: Check if infinite budget and initial budget implementation as here work
        dragonfly_options = {
            "capital_type": "realtime",
            # "max_capital": budget,
            "max_capital": float("inf"),
            "max_or_min": "min",
            "init_capital": budget * init_frac
        }
        options, config = load_dragonfly_options(options=dragonfly_options, config=config)

        if options.max_capital < 0:
            raise ValueError('max_capital (time or number of evaluations) must be positive.')

        is_mf = fidelity_parsers is not None

        def parse_domain(x):
            return Configuration(
                configuration_space=self.cs,
                values={parser[0]: parser[1](val) for parser, val in zip(domain_parsers, x)}
            )

        def objective(x):
            return self.benchmark.objective_function(parse_domain(x))['function_value']

        # Change the current working directory to a unique temporary location in order to avoid any
        # huge messes due to dragonfly's multi-process communication system

        old_cwd = Path().cwd()
        change_cwd()

        if is_mf:
            def parse_fidelities(z):
                ret = {parser[0]: parser[1](val) for parser, val in zip(fidelity_parsers, z)}
                return ret

            def objective_mf(z, x):
                conf = parse_domain(x)
                fidels = parse_fidelities(z)
                logger.debug("Calling multi-fidelity objective with configuration %s at fidelity %s" % (
                    conf.get_dictionary(), fidels))
                ret = self.benchmark.objective_function(conf, fidelity=fidels)
                logger.debug("multi-fidelity objective returned %s" % ret)
                return ret['function_value']

            def cost(z):
                ret = sum([c(v) for c, v in zip(fidelity_costs, z)]) / float(len(fidelity_costs)) + 1e-6
                return ret

            logger.info('Minimising multi-fidelity function on\n Fidelity-Space: %s.\n Domain: %s.' % (
                config.fidel_space, config.domain))
            opt_val, opt_pt, history = minimise_multifidelity_function(
                objective_mf, fidel_space=None, domain=None,
                fidel_to_opt=config.fidel_to_opt, fidel_cost_func=cost,
                max_capital=options.max_capital, capital_type=options.capital_type,
                opt_method=options.opt_method, config=config, options=options,
                reporter=options.report_progress)
        else:
            # test_obj = objective(test_point)
            logger.info('Minimising function on Domain: %s.' % config.domain)
            opt_val, opt_pt, history = minimise_function(
                objective, domain=None, max_capital=options.max_capital,
                capital_type=options.capital_type, opt_method=options.opt_method,
                config=config, options=options, reporter=options.report_progress)

        # Go back to the original working directory we started from
        os.chdir(old_cwd)

        generate_trajectory(history, save_file=self.output_dir / f"dragonfly_traj.json",
                            is_cp=True if isinstance(history.query_qinfos[0].point, list) else False,
                            history_file=self.output_dir / "dragonfly_history.json")

        # return self.optimizer_settings['output_dir']
