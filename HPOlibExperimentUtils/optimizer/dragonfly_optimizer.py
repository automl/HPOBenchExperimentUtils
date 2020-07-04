import logging
from pathlib import Path

from HPOlibExperimentUtils.optimizer.base_optimizer import Optimizer

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

        # Ok following
        # https://stackoverflow.com/questions/2837214/python-popen-command-wait-until-the-command-is-finished
        # call is the command to use.
        import subprocess
        subprocess.call(['ls -l', '>', 'text.txt'])

        # or it is in the output folder then simply self.optimizer_settings['output_dir']
        return Path('The Folder where your trajectory is')
