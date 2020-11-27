plot_dc = {
    "BNNOnBostonHousing": {
    # BOHB paper
    "xlim_lo": 10**2,
    "ylim_lo": 3,
    "ylim_up": 8, #70,
    "xscale": "log",
    "yscale": "linear",
    # None yet
    "ystar_valid": 0,
    "ystar_test": 0,
    },
    "BNNOnProteinStructure": {
    "xlim_lo": 10**2,
    "ylim_lo": 3,
    "ylim_up": 5, #9,
    "xscale": "log",
    "yscale": "linear",
    # None yet
    "ystar_valid": 0,
    "ystar_test": 0,
    },
    "BNNOnYearPrediction": {
        "xlim_lo": 10**2,
        "ylim_lo": 2,
        "ylim_up": 40, #50,
        "xscale": "log",
        "yscale": "linear",
        # None yet
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "cartpolereduced": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**2,
        "ylim_up": 10**4,
        "xscale": "log",
        "yscale": "log",
        # None yet
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "SliceLocalizationBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-8,
        "ylim_up": 10**-1,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
        # import numpy as np
        # bench = SliceLocalizationBenchmark(rng=1, data_path=<path>)
        #
        # configs, te, ve = [], [], []
        # for k in bench.benchmark.data.keys():
        #     t = bench.benchmark.data[k]["final_test_error"]
        #     t = [float(i) for i in t]
        #     te.append(np.sum(t) / len(t))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.00019159916337230243, #0.00020406871,
        "ystar_test": 0.00014428208305616863,
    },
    "ProteinStructureBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-6,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.tabular_benchmarks import ProteinStructureBenchmark
        # import numpy as np
        # bench = ProteinStructureBenchmark(rng=1, data_path=<path>)
        #
        # configs, te, ve = [], [], []
        # for k in bench.benchmark.data.keys():
        #     t = bench.benchmark.data[k]["final_test_error"]
        #     t = [float(i) for i in t]
        #     te.append(np.sum(t) / len(t))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.221378855407238, #0.22137885,
        "ystar_test": 0.21536805480718613,
    },
    "NavalPropulsionBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-9,
        "ylim_up": 10**-1,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.tabular_benchmarks import NavalPropulsionBenchmark
        # import numpy as np
        # bench = NavalPropulsionBenchmark(rng=1, data_path=<path>)
        #
        # configs, te, ve = [], [], []
        # for k in bench.benchmark.data.keys():
        #     t = bench.benchmark.data[k]["final_test_error"]
        #     t = [float(i) for i in t]
        #     te.append(np.sum(t) / len(t))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 3.19113473778998e-05, #3.1911346e-05,
        "ystar_test": 2.9110290597600397e-05,
    },
    "ParkinsonsTelemonitoringBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-7,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.tabular_benchmarks import ParkinsonsTelemonitoringBenchmark
        # import numpy as np
        # bench = ParkinsonsTelemonitoringBenchmark(rng=1, data_path=<path>)
        #
        # configs, te, ve = [], [], []
        # for k in bench.benchmark.data.keys():
        #     t = bench.benchmark.data[k]["final_test_error"]
        #     t = [float(i) for i in t]
        #     te.append(np.sum(t) / len(t))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.0067059280117973685, #0.007629349,
        "ystar_test": 0.004239296889863908,
    },
    "NASCifar10ABenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-6,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py#L26
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
    "NASCifar10BBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py#L26
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
    "NASCifar10CBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py#L26
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
   "Cifar100NasBench201Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        #  cifar100           train       (9930, 99.93733333333334)
        # cifar100            x-valid     (9930, 73.4933333577474)    obj_func
        # cifar100            x-test      (9930, 73.51333332112631)
        # cifar100            ori-test    (9930, 73.50333333333333)   obj_func_test
        "ystar_valid": 26.5066666422526,
        "ystar_test": 26.49666666666667,
   },
   "Cifar10ValidNasBench201Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        # cifar10-valid       train       (10154, 100.0)
        # cifar10-valid       x-valid     (6111, 91.60666665039064)   obj_func
        # cifar10-valid       ori-test    (1459, 91.52333333333333)   obj_func_test
        # [27.11.2020] Note: Adding one more digit for ystar_valid
        "ystar_valid": 8.393333349609364,
        "ystar_test": 8.47666666666667,
   },
   "Cifar10NasBench201Benchmark":  {
        "xlim_lo": 10**2,
        "ylim_lo": 5,
        "ylim_up": 20,
        "xscale": "log",
        "yscale": "log",
        # cifar10             train       (10484, 99.994)
        # cifar10             ori-test    (6111, 94.37333333333333)   both
   },
   "ImageNetNasBench201Benchmark":   {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        # ImageNet16-120      train       (9930, 73.22918040138735)
        # ImageNet16-120      x-valid     (10676, 46.73333327229818)  obj_func
        # ImageNet16-120      x-test      (857, 47.31111100599501)
        # ImageNet16-120      ori-test    (857, 46.8444444647895)     obj_func_test
        "ystar_valid": 53.26666672770182,
        "ystar_test": 53.1555555352105,
   },
}

list_of_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']

color_per_opt = {
    "hpbandster_bohb_eta_3": list_of_colors[0],
    "hpbandster_bohb_eta_2": list_of_colors[0],
    "smac_hb_eta_3": list_of_colors[1],
    "smac_hb_eta_2": list_of_colors[1],
    "randomsearch": list_of_colors[2],
    "dragonfly_default": list_of_colors[3],
    "dehb": list_of_colors[4],
}

marker_per_opt = {
    "hpbandster_bohb_eta_3": "o",
    "hpbandster_bohb_eta_2": "o",
    "smac_hb_eta_3": "s",
    "smac_hb_eta_2": "s",
    "randomsearch": "v",
    "dragonfly_default": "^",
    "dehb": "*",
}
