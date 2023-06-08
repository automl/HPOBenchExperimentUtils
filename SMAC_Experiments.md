# Installation and Setup

```bash
# Create conda environment
conda create -n hpobexp python=3.10 -y
conda activate hpobexp

# Clone and install HPOBench experiment utils
git clone https://github.com/automl/HPOBenchExperimentUtils.git
cd HPOBenchExperimentUtils
pip install -e .
cd ..

# Clone and install HPOBench 
git clone https://github.com/automl/HPOBench.git
cd HPOBench
pip install -e .
cd ..

# Create the Hpobenchrc file
python -c "import hpobench; print(hpobench.__version__)"
# Check it with
vim ~/.config/hpobench/.hpobenchrc

# Install the optimizers
# For reproducing the SMAC journal paper results we only need the ones below from [autogluon,dehb,dragonfly,hpbandster,optuna,smac,ray_base,ray_hyperopt,ray_bayesopt]
pip install -e "HPOBenchExperimentUtils/.[dragonfly,hpbandster,smac]"


cd HPOBenchExperimentUtils

# ADAPT STARTUP.SH
# Edit the singularity path
# You can find the path with `which singularity`.
# If singularity is not installed on your machine, you can install it via `bash ci_scripts/install_singularity.sh`.
vim scripts/startup.sh
```

# Run Experiments
```bash
# Run experiments with
bash runcommands/run_SMAC_experiments.cmd
```

SMAC uses following benchmarks:

- paramnettime: ParamNetLetterOnTimeBenchmark
- NASTABORIG: NavalPropulsionBenchmarkOriginal
- NAS1SHOT1: NASBench1shot1SearchSpace2Benchmark

The runcommands for the benchmarks were generated with 
```bash
python scripts/create_cmd.py --exp paramnettime --opt smac --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NASTABORIG --opt smac  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NAS1SHOT1 --opt smac  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp paramnettime --opt dragonfly  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NASTABORIG --opt dragonfly  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NAS1SHOT1 --opt dragonfly  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp paramnettime --opt hpband  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NASTABORIG --opt hpband  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NAS1SHOT1 --opt hpband  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp paramnettime --opt rs  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NASTABORIG --opt rs  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
python scripts/create_cmd.py --exp NAS1SHOT1 --opt rs  --out-cmd runcommands --python-cmd sudo ~/mambaforge/envs/hpobexp/bin/python
```

# Troubles?
Something does not work while running?
Add the --debug flag to a single command.

## pyro4 communication error
Maybe you don't have enough permissions to communicate.
Then you could try regenerating the commands with the `--python-cmd` flag, like so:
`python scripts/create_cmd.py --exp paramnettime --opt smac --out-cmd runcommands --python-cmd sudo python ~/conda/envs/yourenv/bin/python`

## Singularity instance does not start 
Maybe you do not have enough rights.
Add `sudo` in front of every singularity command in `HPOBench/hpobench/container/client_abstract_benchmark.py`.
