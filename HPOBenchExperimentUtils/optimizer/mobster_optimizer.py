import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Union
import shutil
import time
import types

try:
    import autogluon.core as ag
    from autogluon.core.scheduler.hyperband import _get_rung_levels
    import autogluon.core.version as v
    assert v.__version__ == "0.0.15b20201205"
except:
    # We need to use this specific version since changes introduces later on won't work with our
    # setup (https://github.com/awslabs/autogluon/commit/21483d9b9c3b7c5da935e27c46f2a52e0471bf54)
    raise ValueError("""
    Autogluon is not installed or the wrong version is installed, please run:\n
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade setuptools
    python3 -m pip install --upgrade "mxnet<2.0.0"
    python3 -m pip install autogluon==0.0.15b20201205
    """)

try:
    import dill
    raise ValueError("`dill` is installed, please remove it via:\npip uninstall dill")
except ModuleNotFoundError:
    pass


from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer

_log = logging.getLogger(__name__)


def nothing(self):
    print("Do not stop container")


class MobSterOptimizer(SingleFidelityOptimizer):
    """
    This class implements a random search optimizer.
    All benchmark and optimizer specific information are stored in the dictionaries
    benchmark_settings and optimizer_settings.
    """
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        # Setup can be done here or in run()
        _log.info('Successfully initialized')
        self.done = False

        # Set up search space and rung_levels
        self.ag_space = self._get_ag_space()
        if isinstance(self.main_fidelity, UniformFloatHyperparameter):
            raise ValueError("Mobster can't handle float fidelities")

        # Get time limit
        self.time_limit = self.benchmark.wall_clock_limit_in_s
        if self.benchmark.is_surrogate:
            # We limit this to 24h
            self.time_limit = 60*60*24

        # Get Mobster settings
        self.reduction_factor = self.settings["reduction_factor"]
        self.searcher = self.settings.get("searcher", "bayesopt")
        assert self.searcher in ("bayesopt", "random"), self.searcher
        self.scheduler = self.settings.get("scheduler", "hyperband_promotion")
        assert self.scheduler in ["hyperband_promotion", "hyperband_stopping"]
        self.hyperband_type = None
        if self.scheduler == "hyperband_stopping":
            self.hyperband_type = "stopping"
        elif self.scheduler == "hyperband_promotion":
            self.hyperband_type = "promotion"

        # Some fixed settings
        # setting the number of brackets to 1 means that we run effectively successive halving
        self.brackets = int(self.settings.get("brackets", 1))
        self.first_is_default = False
        self.num_gpus = 0
        self.num_cpus = 1
        self.rung_levels = self._get_rung_levels()

        # Autogluon evaluates every configuration in its own subprocess and thus would
        # terminate the container after evaluation. Here, we overwrite the shutdown method.
        self.benchmark.benchmark._real_shutdown = self.benchmark.benchmark._shutdown
        self.benchmark.benchmark._shutdown = types.MethodType(nothing, self.benchmark.benchmark)

    def setup(self):
        pass

    def _get_rung_levels(self):
        # According to line 820 in hyperband.py
        rung_levels = _get_rung_levels(rung_levels=None, grace_period=self.min_budget,
                                       reduction_factor=self.reduction_factor,
                                       max_t=self.max_budget)
        rung_levels = rung_levels + [self.max_budget]
        return rung_levels

    def _get_ag_space(self):
        d = dict()
        for h in self.cs.get_hyperparameters():
            if isinstance(h, UniformFloatHyperparameter):
                d[h.name] = ag.space.Real(lower=h.lower, upper=h.upper, log=h.log)
            elif isinstance(h, UniformIntegerHyperparameter):
                if h.log:
                    # autogluon does not support int with log=True, make this a Real and round
                    d[h.name] = ag.space.Real(lower=h.lower, upper=h.upper, log=True)
                else:
                    d[h.name] = ag.space.Int(lower=h.lower, upper=h.upper, log=False)
            elif isinstance(h, CategoricalHyperparameter):
                d[h.name] = ag.space.Categorical(*h.choices)
            elif isinstance(h, OrdinalHyperparameter):
                # autogluon does not support ordinals, make this an Int
                d[h.name] = ag.space.Int(lower=0, upper=len(h.sequence)-1)
            else:
                raise ValueError("Cannot handle %s" % h.name)
        return d

    def make_benchmark(self):
        #@ag.args(**self.ag_space, epochs=mb, valid_budgets=rl)
        def objective_function(args, reporter):
            # Get time used until now
            config = dict()
            for h in args.cs.get_hyperparameters():
                if isinstance(h, UniformIntegerHyperparameter):
                    config[h.name] = np.round(args[h.name])
                elif isinstance(h, OrdinalHyperparameter):
                    config[h.name] = h.sequence[int(args[h.name])]
                else:
                    config[h.name] = args[h.name]
            for epoch in args.valid_budgets:
                fidelity = {args.main_fidelity.name: epoch}
                res = self.benchmark.objective_function(config, fidelity=fidelity)
                # Autogluon maximizes, HPOBench returns something to be minimized
                acc = -res['function_value']
                eval_time = res['cost']
                reporter(
                    epoch=epoch,
                    performance=acc,
                    eval_time=eval_time,
                    time_step=time.time(), **config)

        return ag.args(**self.ag_space, epochs=self.max_budget, valid_budgets=self.rung_levels,
                       cs=self.cs,
                       main_fidelity=self.main_fidelity)(objective_function)

    def _fix_runhistory(self):
        if not self.done:
            raise ValueError("Optimization not yet finished")
        dest = Path.joinpath(self.benchmark.log_file.parent,
                             self.benchmark.log_file.name + ".ORIGINAL")
        shutil.move(self.benchmark.log_file, dest)

        last_stamp = None
        total_time_used = 0
        total_objective_costs = 0
        function_call = 0
        with open(dest, "r") as fh:
            for line in fh:
                record = json.loads(line)
                if "boot_time" in record:
                    self.benchmark.write_line_to_file(file=self.benchmark.log_file,
                                                      dict_to_store=record)
                    start = record["boot_time"]
                    last_stamp = start
                    continue
                total_time_used += (record["start_time"] - last_stamp)
                total_time_used += record["cost"]
                total_objective_costs += record["cost"]
                function_call += 1

                # Check whether any of these exceeds limit
                if self.benchmark.tae_limit and (function_call > self.benchmark.tae_limit) or \
                    total_time_used > self.benchmark.wall_clock_limit_in_s:
                    _log.critical("Used resources exceed limit, stop fixing runhistory")
                    break

                # Overwrite values in record
                record['function_call'] = function_call
                record['total_time_used'] = total_time_used
                record['total_objective_costs'] = total_objective_costs
                self.benchmark.write_line_to_file(file=self.benchmark.log_file,
                                                  dict_to_store=record)

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """
        callback = None
        scheduler = ag.scheduler.HyperbandScheduler(self.make_benchmark(),
                                                    resource={'num_cpus': self.num_cpus,
                                                              'num_gpus': self.num_gpus},
                                                    # Autogluon runs until it either reaches num_trials or time_out
                                                    num_trials=self.benchmark.tae_limit,
                                                    time_out=self.time_limit,
                                                    # This argument defines the metric that will be maximized.
                                                    # Make sure that you report this back in the objective function.
                                                    reward_attr='performance',
                                                    # The metric along we make scheduling decision. Needs to be also
                                                    # reported back to AutoGluon in the objective function.
                                                    time_attr='epoch',
                                                    brackets=self.brackets,
                                                    checkpoint=None,
                                                    training_history_callback=callback,
                                                    training_history_callback_delta_secs=1,
                                                    searcher=self.searcher,
                                                    # Defines searcher for new configurations
                                                    # training_history_callback_delta_secs=args.store_results_period,
                                                    reduction_factor=self.reduction_factor,
                                                    type=self.hyperband_type,
                                                    # rung_levels = [2, 7, 22, 66, 199],
                                                    search_options={'random_seed': self.rng,
                                                                    'first_is_default': self.first_is_default},
                                                    # defines the minimum resource level for Hyperband,
                                                    # i.e the minimum number of epochs
                                                    grace_period=self.min_budget)
        assert scheduler.terminator.rung_levels == self.rung_levels[:-1], \
            (scheduler.terminator.rung_levels, self.rung_levels[:-1])
        scheduler.run()
        scheduler.join_jobs()

    def shutdown(self):
        # Autogluon is done. Shutdown container
        _log.critical("Stop container")
        self.benchmark.benchmark._real_shutdown()
        self.done = True
        # Rewrite trajectory
        self._fix_runhistory()
        _log.info("Suceeded rewriting trajectory")