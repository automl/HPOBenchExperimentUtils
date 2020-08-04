import logging
from pathlib import Path

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOlibExperimentUtils.utils.dragonfly_utils import \
    configspace_to_dragonfly, load_dragonfly_options, generate_trajectory
from HPOlibExperimentUtils.utils import Constants

from dragonfly import minimise_function, \
    minimise_multifidelity_function

from ConfigSpace import Configuration

logger = logging.getLogger('Optimizer')


class DragonflyOptimizer(Optimizer):

    def __init__(self, benchmark, optimizer_settings, benchmark_settings, intensifier, rng=0):
        super().__init__(benchmark, optimizer_settings, benchmark_settings, intensifier, rng)

    def setup(self):
        pass

    def run(self) -> Path:
        """

        Returns
        -------
        Path-object. Please return the path to directory where your trajectory lies. This path is then used to read in
        the trajecotory and transform it to a uniform format. (If you have such a trajectory example for me,
        i can do the further steps then.)
        """

        # TODO: Update to include constraints
        # TODO: Update to use ConfigurationSpace objects for fidelities
        # TODO: Include usage of RNG for consistency
        known_fidelities = ['n_estimators', 'subsample']
        # fidelities = {key: value for key, value in self.benchmark_settings.items()
        #               if key not in Constants.fixed_benchmark_settings}
        fidelities = {key: value for key, value in self.benchmark_settings.items()
                      if key in known_fidelities}
        logger.debug(f"Using fidelities: %s" % fidelities)
        config, domain_parsers, fidelity_parsers = configspace_to_dragonfly(domain_cs=self.cs, fidely_cs=fidelities)
        fidelity_costs = [tup[2] for tup in fidelity_parsers]  # Separate the costs
        fidelity_parsers = [(tup[0], tup[1]) for tup in fidelity_parsers]

        self.optimizer_settings["max_or_min"] = "min"
        options, config = load_dragonfly_options(options=self.optimizer_settings, config=config)

        if options.max_capital < 0:
            raise ValueError('max_capital (time or number of evaluations) must be positive.')

        if hasattr(config, 'fidel_space'):
            logger.debug("Using multi-fidelity optimization.")
            is_mf = True
        else:
            logger.debug("Using single-fidelity optimization.")
            is_mf = False

        def parse_domain(x):
            return Configuration(
                configuration_space=self.cs,
                values={parser[0]: parser[1](val) for parser, val in zip(domain_parsers, x)}
            )

        def objective(x):
            return self.benchmark.objective_function(parse_domain(x))['function_value']

        if is_mf:
            def parse_fidelities(z):
                ret = {parser[0]: parser[1](val) for parser, val in zip(fidelity_parsers, z)}
                return ret

            def objective_mf(z, x):
                conf = parse_domain(x)
                fidels = parse_fidelities(z)
                logger.debug("Calling multi-fidelity objective with configuration %s at fidelity %s" % (
                    conf.get_dictionary(), fidels))
                ret = self.benchmark.objective_function(conf, **fidels)
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

        generate_trajectory(history, save_file=self.optimizer_settings["output_dir"] / Constants.smac_traj_filename,
                            is_cp=True if isinstance(history.query_qinfos[0].point, list) else False,
                            history_file=self.optimizer_settings["output_dir"] / "saved_history.json")

        # Ok following
        # https://stackoverflow.com/questions/2837214/python-popen-command-wait-until-the-command-is-finished
        # call is the command to use.
        # import subprocess
        # subprocess.call(['ls -l', '>', 'text.txt'])

        return self.optimizer_settings['output_dir']
