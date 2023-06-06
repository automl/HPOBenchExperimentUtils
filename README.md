HPOBenchExperimentUtils
---------------------

A small tool to easily run different optimizers on HPOBench-benchmarks with the same settings. 
The HPOBenchExpUtils extract for each run a runhistory as well as a trajectory. 

## SMAC Journal Experiments
For reproducing the experiments in our [SMAC JMLR article](https://jmlr.org/papers/v23/21-0888.html), please check out `SMAC_Experiments.md`.


## Running a benchmark

The hpo run can be started from either the commandline:
```python
from HPOBenchExperimentUtils import run_benchmark
run_benchmark(optimizer='hpbandster_bohb_eta_3',
              benchmark='cartpolereduced',
              output_dir='path/to/output',
              rng=0)
``` 

or by using the commandline:

```bash 
python run_benchmark.py --output_dir path/to/output \
                        --optimizer smac_hb_eta_2 \
                        --benchmark xgboost \
                        --task_id 167083 \
                        --rng 1
```

The tool automatically saves the evaluated configurations as well as the seen trajectory. 
Both files are stored in the output_dir/<optimizer_string>-run-<rng>. Each line in both files is a json dict, which 
stores information about the evaluated configuration.  

**Note** that in both cases you can pass benchmark specific parameters to the call. Here, the xgboost benchmark takes an
openml task id. Please take a look at the benchmarks in the 
[HPOBench](https://github.com/automl/HPOBench/tree/master/hpobench/benchmarks).
Also, by default the containerized version of the benchmark is used. This requires singularity 3.5. You can use the 
local installed benchmarks by adding use_local=True to the function call. 

## Validating configurations
The HPOBenchExperimentUtils tool also validates previously found trajectories. Validating means running the 
configuration again but this time on the test-objective function of the benchmark with the highest budget. This step can
take a lot of time. 

The tool reads all configurations found in the specified path and valdiates them. 

Call the validation function again either from code:

```python
from HPOBenchExperimentUtils import validate_benchmark
validate_benchmark(benchmark='cartpolereduced',
                   output_dir='path / to / output',
                   rng=0)
``` 

... or the commandline:

```bash 
python validate_benchmark.py --output_dir path/to/output \
                             --benchmark xgboost \
                             --task_id 167083 \
                             --rng 1
```
The validated trajectory is automatically saved in human readable form to the output directory. 

## Settings

The benchmarks' settings are predefined in the file 
[benchmark_settings.yaml](./HPOBenchExperimentUtils/benchmark_settings.yaml). The settings for the optimizer including 
timelimits and cutoff times are defined in the 
[optimizer_settings.yaml](./HPOBenchExperimentUtils/optimizer_settings.yaml)

### Available Optimizer settings
| Optimizer                  	| Available options                                                          	|
|----------------------------	|----------------------------------------------------------------------------	|
| SMAC - Hyperband           	| smac_hb_eta_1, smac_hb_eta_2_learna, smac_hb_eta_2, smac_hb_eta_3          	|
| SMAC - Successive Halving  	| smac_sh_eta_1, smac_sh_eta_2_learna, smac_sh_eta_2, smac_sh_eta_3          	|
| HpBandSter - BOHB          	| hpbandster_bohb_eta_2_learna, hpbandster_bohb_eta_2, hpbandster_bohb_eta_3 	|
| HpBandSter - Random Search 	| hpbandster_rs_eta_2_learna, hpbandster_rs_eta_2, hpbandster_rs_eta_3       	|
| HpBandSter - Hyperband     	| hpbandster_hb_eta_2_learna, hpbandster_hb_eta_2, hpbandster_hb_eta_3       	|
| HpBandSter - H2BO          	| hpbandster_h2bo_eta_2_learna, hpbandster_h2bo_eta_2, hpbandster_h2bo_eta_3 	|
| Dragonfly                   	| dragonfly_default, dragonfly_realtime                                         |

### Available Benchmarks:
| Benchmarks                                    	| benchmark token                   	| HPOBench Link                                                                                      	|
|-----------------------------------------------	|-----------------------------------	|---------------------------------------------------------------------------------------------------	|
| Cartpole - Full search space                  	| cartpolefull                      	| Link                                                                                              	|
| Cartpole - Reduced search space               	| cartpolereduced                   	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/rl/cartpole.py)            	|
| Learna                                        	| learna                            	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/rl/learna_benchmark.py)    	|
| MetaLearna                                    	| metalearna                        	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/rl/learna_benchmark.py)    	|
| NasBench101 - Cifar10A                        	| NASCifar10ABenchmark              	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/nasbench_101.py)       	|
| NasBench101 - Cifar10B                        	| NASCifar10BBenchmark              	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/nasbench_101.py)       	|
| NasBench101 - Cifar10C                        	| NASCifar10CBenchmark              	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/nasbench_101.py)       	|
| TabularBenchmarks - Naval Propulsion          	| NavalPropulsionBenchmark          	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/tabular_benchmarks.py) 	|
| TabularBenchmarks - Parkinsons Telemonitoring 	| ParkinsonsTelemonitoringBenchmark 	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/tabular_benchmarks.py) 	|
| TabularBenchmarks - Protein Structure         	| ProteinStructureBenchmark         	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/tabular_benchmarks.py) 	|
| TabularBenchmarks - Slice Localization        	| SliceLocalizationBenchmark        	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/nas/tabular_benchmarks.py) 	|
| XGBoost Benchmark                             	| xgboost                           	| [link](https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/ml/xgboost_benchmark.py)   	|

## How to contribute:

### New Benchmark Settings:
If you want to add a new benchmark setting, add a yaml conform entry in the 
[benchmark_settings.yaml](./HPOBenchExperimentUtils/benchmark_settings.yaml)

Possible options are:
```yaml 

xgboost:
  # Mandatory options:
  # ##################
  time_limit_in_s: 4000
  cutoff_in_s: 1800
  mem_limit_in_mb: 4000
  
  # Address in the hpobench
  import_from: ml.xgboost_benchmark
  import_benchmark: XGBoostBenchmark
  

  # Facultative options: (Only need to be specified if used)
  # ####################
  # If the benchmark has multiple fidelities, you can specify a main fidelity. This then used by the 
  # SingleFidelityOptimizer.
  main_fidelity: subsample

  # If the benchmark is a surrogate (e.g. Nasbench201 is a tabular benchmark), please set the option to true. 
  # By default, this option is set to false. This option changes the remaining budget calculation in the bookkeeper.
  is_surrogate: true
```

### New Optimizer Settings:
Analogously to the benchmark settings, you can add a new optimizer setting to the 
[optimizer_settings.yaml](./HPOBenchExperimentUtils/optimizer_settings.yaml).

```yaml
# The name of the optimizer setting can be chosen freely.
hpbandster_hb_eta_3_test:
  # Specifies the optimizer to use. See table below for supported optimizers
  optimizer: hpbandster_hb
  
  # Optimizer dependent options:
  # ############################
  eta: 3
```

#### Available Optimizers:
| Optimizer                  | optimizer string |
|----------------------------|------------------|
| SMAC - Hyperband           | smac_hb          |
| SMAC - Successive Halving  | smac_sh          |
| HpBandSter - BOHB          | hpbandster_bohb  |
| HpBandSter - Random Search | hpbandster_rs    |
| HpBandSter - Hyperband     | hpbandster_hb    |
| HpBandSter - H2BO          | hpbandster_h2bo  |

### Add new optimizer: 
- Inherit from the Base Optimizer
- Implement the run method. 
- Add a optimizer setting to the optimizer_settings.yaml as described above. 
It's as simple as that :wink:

### Some optimizer-specific settings:
| Optimizer                 | setting name          | Description                                                   |
|---------------------------|-----------------------|---------------------------------------------------------------|
| Dragonfly                 | init_iter_per_dim     | An integer N such that, given that the benchmark's configuration space has D dimensions, NxD iterations will be used to randomly sample configurations to warm-start the optimizer with. Makes dragonfly use an internal budget type of 'num_evals'. |
| Dragonfly                 | init_capital_frac     | A value f in the closed interval [0, 1] such that, given that a benchmark specifies a time limit of T seconds, f * t seconds will be used for initialization. Only comes into effect when 'init_iter_per_dim' is not given. Also switches dragonfly's internal budget type to 'realtime'. |
