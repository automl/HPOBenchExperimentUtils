plot_dc = {
    "BNNOnBostonHousing": {
    # BOHB paper
    "xlim_lo": 10**2,
    "ylim_lo": 3,
    "ylim_up": 8, #70,
    "xscale": "log",
    "yscale": "linear",
    # None
    "ystar_valid": 0,
    "ystar_test": 0,
    },
    "BNNOnProteinStructure": {
    "xlim_lo": 10**2,
    "ylim_lo": 3,
    "ylim_up": 5, #9,
    "xscale": "log",
    "yscale": "linear",
    # None
    "ystar_valid": 0,
    "ystar_test": 0,
    },
    "BNNOnYearPrediction": {
        "xlim_lo": 10**2,
        "ylim_lo": 2,
        "ylim_up": 40, #50,
        "xscale": "log",
        "yscale": "linear",
        # None
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "cartpolereduced": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**2,
        "ylim_up": 10**4,
        "xscale": "log",
        "yscale": "log",
        # None
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
        #     te.append(np.mean(bench.benchmark.data[k]["final_test_error"]))
        #     ve.append(np.mean(bench.benchmark.data[k]["valid_mse"][:, -1]))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.00019159916, #0.00020406871,
        "ystar_test": 0.00014428208,
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
        #     te.append(np.mean(bench.benchmark.data[k]["final_test_error"]))
        #     ve.append(np.mean(bench.benchmark.data[k]["valid_mse"][:, -1]))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.22137885, #0.22137885,
        "ystar_test": 0.21536806,
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
        #     te.append(np.mean(bench.benchmark.data[k]["final_test_error"]))
        #     ve.append(np.mean(bench.benchmark.data[k]["valid_mse"][:, -1]))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 3.1911346e-05, #3.1911346e-05,
        "ystar_test": 2.9110292e-05,
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
        #     te.append(np.mean(bench.benchmark.data[k]["final_test_error"]))
        #     ve.append(np.mean(bench.benchmark.data[k]["valid_mse"][:, -1]))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0067059286, #0.007629349,
        "ystar_test": 0.004239297,
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
        "ystar_valid": 26.506666642252596,
        "ystar_test": 26.49666666666667,
   },
   "Cifar10ValidNasBench201Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 8.393333349609367,
        "ystar_test": 8.476666666666677,
   },
   "Cifar10NasBench201Benchmark":  {
        "xlim_lo": 10**2,
        "ylim_lo": 5,
        "ylim_up": 20,
        "xscale": "log",
        "yscale": "log",
   },
   "ImageNetNasBench201Benchmark":   {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
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
