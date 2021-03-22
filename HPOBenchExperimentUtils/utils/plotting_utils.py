import matplotlib.pyplot as plt

plot_dc = {
    "BNNOnBostonHousing": {
        # BOHB paper
        "xlim_lo": 10**2,
        "ylim_lo": 3,
        "ylim_up": 8,
        "xscale": "log",
        "yscale": "linear",
        # None yet
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "BNNOnProteinStructure": {
        "xlim_lo": 10**2,
        "ylim_lo": 3,
        "ylim_up": 5,
        "xscale": "log",
        "yscale": "linear",
        # None yet
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "BNNOnYearPrediction": {
        "xlim_lo": 10**2,
        "ylim_lo": 2,
        "ylim_up": 40,
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
        "xlim_lo": 10**0,
        "ylim_lo": 10**-6,
        "ylim_up": 10**0,
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
        "ylim_lo": 10**-4,
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
        "ylim_lo": 10**-6,
        "ylim_up": 10**0,
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
        "ylim_lo": 10**-4,
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
    "NASCifar10ABenchmark_fixed_seed_0": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py#L26
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
    "NASCifar10ABenchmark_random_seed": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py#L26
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
    "NASCifar10ABenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3.5,
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
        "ylim_lo": 10**-1,
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
        "ylim_lo": 10**-2,
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
    "ImageNetNasBench201Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-1,
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
    "NASBench1shot1SearchSpace1Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-2,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # scripts/Nas1shot1_Incumbent.py
        "ystar_valid": 1-0.9471821784973145,
        "ystar_test": 1-0.9420072237650553,

    },
    "NASBench1shot1SearchSpace2Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-2,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # scripts/Nas1shot1_Incumbent.py
        "ystar_valid": 1-0.9456797440846761,
        "ystar_test": 1-0.9396701256434122,

    },
    "NASBench1shot1SearchSpace3Benchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-2,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # scripts/Nas1shot1_Incumbent.py
        "ystar_valid": 1-0.9473824898401896,
        "ystar_test": 1-0.941773513952891,

    },
    "ParamNetAdultOnStepsBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-3,
        "ylim_up": 10**1,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "ParamNetAdultOnTimeBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3,
        "ylim_up": 10**-1,
        "xscale": "log",
        "yscale": "log",
        # from stats json file March 9th
        "ystar_valid": 0.1437777783975557,
        "ystar_test": 0,
    },
    "ParamNetHiggsOnStepsBenchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-5,
        "ylim_up": 10**-1,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "ParamNetHiggsOnTimeBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3,
        "ylim_up": 10**-1,
        "xscale": "log",
        "yscale": "log",
        # from stats json file March 9th
        "ystar_valid": 0.2739252715252642,
        "ystar_test": 0,
    },
    "ParamNetLetterOnStepsBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "ParamNetLetterOnTimeBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # from stats json file March 9th
        "ystar_valid": 0.030800543060296292,
        "ystar_test": 0,
    },
    "ParamNetMnistOnStepsBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-4.5,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "ParamNetMnistOnTimeBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-4,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # from stats json file March 9th
        "ystar_valid": 0.014969998741149893,
        "ystar_test": 0,
    },
    "ParamNetOptdigitsOnStepsBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "ParamNetOptdigitsOnTimeBenchmark": {
        "xlim_lo": 10**0,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # from stats json file March 9th
        "ystar_valid": 0.011504424673266123,
        "ystar_test": 0,
   },
   "ParamNetPokerOnStepsBenchmark": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-4,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0,
        "ystar_test": 0,
    },
    "ParamNetPokerOnTimeBenchmark": {
        "xlim_lo": 10**2,
        "ylim_lo": 10**-4,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        # from stats json file March 9th
        "ystar_valid": 0.00016679192417291545,
        "ystar_test": 0,
    },
    "NASCifar10ABenchmark_fixed_seed_0": {
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py#L26
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
    "NASCifar10ABenchmark_random_seed": {
        # See original benchmark
        "xlim_lo": 10**1,
        "ylim_lo": 10**-3.5,
        "ylim_up": 10**-0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0.04944576819737756,
        "ystar_test": 0.056824247042338016,
    },
    "ProteinStructureBenchmark_fixed_seed_0": {
        # See original benchmark
        "xlim_lo": 10**0,
        "ylim_lo": 10**-4,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0.221378855407238,
        "ystar_test": 0.21536806225776672,
    },
    "ProteinStructureBenchmark_random_seed": {
        # See original benchmark
        "xlim_lo": 10**0,
        "ylim_lo": 10**-4,
        "ylim_up": 10**0,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 0.221378855407238,
        "ystar_test": 0.21536806225776672,    
    },
    "Cifar10ValidNasBench201Benchmark_fixed_seed_777": {
        # See original benchmark
        "xlim_lo": 10**2,
        "ylim_lo": 10**-2,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 8.393333349609364,
        "ystar_test": 8.476666666666674,
    },
    "Cifar10ValidNasBench201Benchmark_random_seed": {
        # See original benchmark
        "xlim_lo": 10**2,
        "ylim_lo": 10**-2,
        "ylim_up": 10**2,
        "xscale": "log",
        "yscale": "log",
        "ystar_valid": 8.393333349609364,
        "ystar_test": 8.476666666666674,    
    },
}

list_of_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']

color_per_opt = {
    "hpbandster_bohb_eta_3": list_of_colors[0],
    "hpbandster_bohb_eta_2": list_of_colors[0],
    "hpbandster_hb_eta_3": list_of_colors[7],
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
    "hpbandster_hb_eta_3": "o",
    "smac_hb_eta_3": "s",
    "smac_hb_eta_2": "s",
    "smac_sf": "s",
    "randomsearch": "v",
    "dragonfly_default": "^",
    "dehb": "*",
    "autogluon": "o",
}


def unify_layout(ax, fontsize=15, legend_args=None, title=None):
    if legend_args is None:
        legend_args = {}
    ax.legend(fontsize=fontsize, **legend_args)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.get_label().set_fontsize(fontsize)
    ax.yaxis.get_label().set_fontsize(fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.grid(b=True, which="both", axis="both", alpha=0.5)


benchmark_families = {
    "NAS201": ["Cifar10ValidNasBench201Benchmark", "Cifar100NasBench201Benchmark",
               "ImageNetNasBench201Benchmark"],
    "NAS101": ["NASCifar10ABenchmark", "NASCifar10BBenchmark", "NASCifar10CBenchmark"],
    "NASTAB": ["SliceLocalizationBenchmark", "ProteinStructureBenchmark",
               "NavalPropulsionBenchmark", "ParkinsonsTelemonitoringBenchmark", ],
    "NAS1SHOT1": ["NASBench1shot1SearchSpace1Benchmark", "NASBench1shot1SearchSpace2Benchmark",
                  "NASBench1shot1SearchSpace3Benchmark", ],
    "pybnn": ["BNNOnBostonHousing", "BNNOnProteinStructure", "BNNOnYearPrediction", ],
    "rl": ["cartpolereduced"],
    "learna": ["metalearna", "learna"],
    "paramnetsteps": ["ParamNetAdultOnStepsBenchmark", "ParamNetHiggsOnStepsBenchmark",
                      "ParamNetLetterOnStepsBenchmark", "ParamNetMnistOnStepsBenchmark",
                      "ParamNetOptdigitsOnStepsBenchmark", "ParamNetPokerOnStepsBenchmark", ],
    "paramnettime": ["ParamNetAdultOnTimeBenchmark", "ParamNetHiggsOnTimeBenchmark",
                     "ParamNetLetterOnTimeBenchmark", "ParamNetMnistOnTimeBenchmark",
                     "ParamNetOptdigitsOnTimeBenchmark", "ParamNetPokerOnTimeBenchmark", ],
}
