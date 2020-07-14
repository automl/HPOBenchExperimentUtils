import logging
from pathlib import Path

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOlibExperimentUtils.utils.dragonfly_utils import \
    configspace_to_dragonfly, load_dragonfly_options, generate_trajectory
from HPOlibExperimentUtils.utils.optimizer_utils import Constants

from dragonfly import minimise_function, \
    minimise_multifidelity_function, \
    multiobjective_minimise_functions

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

        # TODO: Update to include constraints and fidelities
        # TODO: Include usage of RNG for consistency
        config, domain_parser = configspace_to_dragonfly(self.cs)

        fidelities = {key: value for key, value in self.benchmark_settings.items()
                      if key not in Constants.fixed_benchmark_settings}

        parse_domain = lambda x: Configuration(
            configuration_space=self.cs,
            values = {parser[0]: parser[1](val) for parser, val in zip(domain_parser, x)}
        )
        objective = lambda x: \
            self.benchmark.objective_function(parse_domain(x), **fidelities)['function_value']

        self.optimizer_settings["max_or_min"] = "min"
        options, config = load_dragonfly_options(options=self.optimizer_settings, config=config)
        if hasattr(config, 'fidel_space'):
            raise RuntimeWarning("Multi-fidelity support is still under implementation and not yet supported. Ignoring "
                                 "assosciated settings.")

        if options.max_capital < 0:
            raise ValueError('max_capital (time or number of evaluations) must be positive.')

        opt_val, opt_pt, history = minimise_function(
            objective, domain=None, max_capital=options.max_capital,
            capital_type=options.capital_type, opt_method=options.opt_method,
            config=config, options=options, reporter=options.report_progress)

        generate_trajectory(history, save_file=self.optimizer_settings["output_dir"] / Constants.trajectory_filename)

        # Ok following
        # https://stackoverflow.com/questions/2837214/python-popen-command-wait-until-the-command-is-finished
        # call is the command to use.
        # import subprocess
        # subprocess.call(['ls -l', '>', 'text.txt'])

        return self.optimizer_settings['output_dir']
