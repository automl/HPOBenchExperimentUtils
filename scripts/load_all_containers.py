from importlib import import_module
import yaml
import os
import sys

base_dir = "/home/eggenspk/2020_Hpolib2/"
settings_file = "%s/HPOBenchExperimentUtils/" % base_dir
experiment_settings_path = "%s/HPOBenchExperimentUtils/benchmark_settings.yaml" % settings_file
with open(experiment_settings_path, 'r') as fh:
    experiment_settings = yaml.load(fh, yaml.FullLoader)

for key in experiment_settings:
    print("Check %s" % key)
    import_str = 'hpobench.' + 'container.' + 'benchmarks.' + experiment_settings[key]["import_from"]
    module = import_module(import_str)
    benchmark_obj = getattr(module, experiment_settings[key]["import_benchmark"])
    try:
        if key in ("xgboostsub", "svm"):
            benchmark_obj = benchmark_obj(task_id=167149)
        else:
            benchmark_obj = benchmark_obj()
    except:
        print("Something went wrong with %s" % key)
    del benchmark_obj
