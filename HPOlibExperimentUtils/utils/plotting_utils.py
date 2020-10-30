plot_dc = {
    "BNNOnBostonHousing": {
    # BOHB paper
    "xlim_lo": 10**3,
    "ylim_lo": 3,
    "ylim_up": 9,
    "xscale": "log",
    "yscale": "linear",
    },
    "BNNOnProteinStructure": {
    "xlim_lo": 10**3,
    "ylim_lo": 2,
    "ylim_up": 10,
    "xscale": "log",
    "yscale": "linear",
    },
    "BNNOnYearPrediction": {
    "xlim_lo": 10**3,
    "ylim_lo": 10,
    "ylim_up": 50,
    "xscale": "log",
    "yscale": "linear",
    },
    "cartpolereduces": {
    "xlim_lo": 10**1,
    "ylim_lo": 10**2,
    "ylim_up": 10**4,
    "xscale": "log",
    "yscale": "log",
    },
}

list_of_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']

color_per_opt = {
    "hpbandster_bohb_eta_3": list_of_colors[0],
    "hpbandster_bohb_eta_2": list_of_colors[0],
    "smac_hb_eta_3": list_of_colors[1],
    "smac_hb_eta_2": list_of_colors[1],
    "randomsearch": list_of_colors[2],
    "dragonfly_default": list_of_colors[3]
}