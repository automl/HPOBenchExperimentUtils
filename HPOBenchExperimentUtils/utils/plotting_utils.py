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
        "ystar_test": 0.0001442820794181898,
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
        #     # do this to be consistent w/ HPOBench and tabular_nas code
        #     te.append(float(np.mean(t)))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.221378855407238, #0.22137885,
        "ystar_test": 0.21536806225776672,
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
        #     # do this to be consistent w/ HPOBench and tabular_nas code
        #     te.append(float(np.mean(t)))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 3.19113473778998e-05, #3.1911346e-05,
        "ystar_test": 2.91102915070951e-05
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
        #     # do this to be consistent w/ HPOBench and tabular_nas code
        #     te.append(float(np.mean(t)))
        #     v = bench.benchmark.data[k]["valid_mse"][:, -1]
        #     v = [float(i) for i in v]
        #     ve.append(np.sum(v) / len(v))
        #
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        "ystar_valid": 0.0067059280117973685, #0.007629349,
        "ystar_test": 0.00423929700627923,
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
        # from hpobench.benchmarks.nas.nasbench_201 import Cifar100NasBench201Benchmark
        # a = Cifar100NasBench201Benchmark()
        # keys = a.data[(777, 'train_acc1es')].keys()
        #
        # configs, te, ve = [], [], []
        # for k in keys:
        #     t1 = a.data[(777, "valid_acc1es")][k][199]
        #     t2 = a.data[(888, "valid_acc1es")][k][199]
        #     t3 = a.data[(999, "valid_acc1es")][k][199]
        #     te.append(float(100 - np.mean([t1, t2, t3])))
        #     v1 = a.data[(777, "test_acc1es")][k]
        #     v2 = a.data[(888, "test_acc1es")][k]
        #     v3 = a.data[(999, "test_acc1es")][k]
        #     ve.append(float(100 - np.mean([v1, v2, v3])))
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        # print(b, best_test, best_valid)
        "ystar_valid": 26.49666666666667,
        "ystar_test": 26.48666667887369,
   },
   "Cifar10ValidNasBench201Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.nasbench_201 import Cifar10ValidNasBench201Benchmark
        # a = Cifar10ValidNasBench201Benchmark()
        # keys = a.data[(777, 'train_acc1es')].keys()
        #
        # configs, te, ve = [], [], []
        # for k in keys:
        #     t1 = a.data[(777, "valid_acc1es")][k][199]
        #     t2 = a.data[(888, "valid_acc1es")][k][199]
        #     t3 = a.data[(999, "valid_acc1es")][k][199]
        #     te.append(float(100 - np.mean([t1, t2, t3])))
        #     v1 = a.data[(777, "test_acc1es")][k]
        #     v2 = a.data[(888, "test_acc1es")][k]
        #     v3 = a.data[(999, "test_acc1es")][k]
        #     ve.append(float(100 - np.mean([v1, v2, v3])))
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        # print(b, best_test, best_valid)
        "ystar_valid": 8.393333349609364,
        "ystar_test": 8.476666666666674,
   },
   "Cifar10NasBench201Benchmark":  {
        "xlim_lo": 10**2,
        "ylim_lo": 5,
        "ylim_up": 20,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark
        # a = ImageNetNasBench201Benchmark()
        # keys = a.data[(777, 'train_acc1es')].keys()
        #
        # configs, te, ve = [], [], []
        # for k in keys:
        #     t1 = a.data[(777, "valid_acc1es")][k][199]
        #     t2 = a.data[(888, "valid_acc1es")][k][199]
        #     t3 = a.data[(999, "valid_acc1es")][k][199]
        #     te.append(float(100 - np.mean([t1, t2, t3])))
        #     v1 = a.data[(777, "test_acc1es")][k]
        #     v2 = a.data[(888, "test_acc1es")][k]
        #     v3 = a.data[(999, "test_acc1es")][k]
        #     ve.append(float(100 - np.mean([v1, v2, v3])))
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        # print(b, best_test, best_valid)
        "ystar_valid": 5.626666666666665,
        "ystar_test": 5.626666666666665,
   },
   "ImageNetNasBench201Benchmark":   {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        # from hpobench.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark
        # a = ImageNetNasBench201Benchmark()
        # keys = a.data[(777, 'train_acc1es')].keys()
        #
        # configs, te, ve = [], [], []
        # for k in keys:
        #     t1 = a.data[(777, "valid_acc1es")][k][199]
        #     t2 = a.data[(888, "valid_acc1es")][k][199]
        #     t3 = a.data[(999, "valid_acc1es")][k][199]
        #     te.append(float(100 - np.mean([t1, t2, t3])))
        #     v1 = a.data[(777, "test_acc1es")][k]
        #     v2 = a.data[(888, "test_acc1es")][k]
        #     v3 = a.data[(999, "test_acc1es")][k]
        #     ve.append(float(100 - np.mean([v1, v2, v3])))
        # best_test = np.min(te)
        # best_valid = np.min(ve)
        # print(b, best_test, best_valid)
        "ystar_valid": 53.1555555352105,
        "ystar_test": 52.68888899400499,
   },
}

list_of_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']

color_per_opt = {
    "hpbandster_bohb_eta_3": list_of_colors[0],
    "hpbandster_bohb_eta_2": list_of_colors[0],
    "smac_hb_eta_3": list_of_colors[1],
    "smac_hb_eta_2": list_of_colors[1],
    "smac_sf": list_of_colors[6],
    "randomsearch": list_of_colors[2],
    "dragonfly_default": list_of_colors[3],
    "dehb": list_of_colors[4],
    "autogluon": list_of_colors[5],
}

marker_per_opt = {
    "hpbandster_bohb_eta_3": "o",
    "hpbandster_bohb_eta_2": "o",
    "smac_hb_eta_3": "s",
    "smac_hb_eta_2": "s",
    "smac_sf": "s",
    "randomsearch": "v",
    "dragonfly_default": "^",
    "dehb": "*",
    "autogluon": "o",
}
