import logging
from pathlib import Path

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer
from HPOlibExperimentUtils.utils import dragonfly_utils

logger = logging.getLogger('Optimizer')


class DragonflyOptimizer(Optimizer):

    def __init__(self, benchmark, optimizer_settings, benchmark_settings, intensifier, rng=0):
        super().__init__(benchmark, optimizer_settings, benchmark_settings, intensifier, rng)

    def setup(self):
        pass

    def run(self) -> Path:
        """
        TODO: DRAGONFLY - This is the skeleton for the dragonfly optimizer.

        Returns
        -------
        Path-object. Please return the path to directory where your trajectory lies. This path is then used to read in
        the trajecotory and transform it to a uniform format. (If you have such a trajectory example for me,
        i can do the further steps then.)
        """

        # TODO: Update to include constraints and fidelities
        from dragonfly import maximise_function, minimise_function
        config, domain_parser = dragonfly_utils.configspace_to_dragonfly(self.cs)

        fidelities = {}
        benchmark_kwargs = {}

        parse_domain = lambda x: {parser[0]: parser[1](val) for parser, val in zip(domain_parser, x)}
        objective = lambda x: \
            self.benchmark.objective_function(parse_domain(x), **fidelities, **benchmark_kwargs)['function_value']




        # Ok following
        # https://stackoverflow.com/questions/2837214/python-popen-command-wait-until-the-command-is-finished
        # call is the command to use.
        import subprocess
        subprocess.call(['ls -l', '>', 'text.txt'])

        # or it is in the output folder then simply self.optimizer_settings['output_dir']
        return Path('The Folder where your trajectory is')
