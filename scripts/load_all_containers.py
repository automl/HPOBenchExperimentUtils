from importlib import import_module
import yaml
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
settings_file = "%s/HPOBenchExperimentUtils/" % base_dir
experiment_settings_path = "%s/HPOBenchExperimentUtils/benchmark_settings.yaml" % settings_file
with open(experiment_settings_path, 'r') as fh:
    experiment_settings = yaml.load(fh, yaml.FullLoader)

for key in experiment_settings:
    import_str = 'hpobench.' + 'container.' + 'benchmarks.' + experiment_settings[key]["import_from"]
    module = import_module(import_str)
    benchmark_obj = getattr(module, experiment_settings[key]["import_benchmark"])
    if key in ("xgboost", "svm"):
        benchmark_obj = benchmark_obj(task_id=167149)
    else:
        benchmark_obj = benchmark_obj()

    del benchmark_obj