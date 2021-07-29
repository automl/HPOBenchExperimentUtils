#### HowTo run experiments on kislurm

### 1) Create Workspace

    # replace <yourname> with your tf-abbreviation, e.g. eggenspk
    ws_allocate HPOBENCH 180 -r 14 -m <yourname>@cs.uni-freiburg.de
    ws_send_ical HPOBENCH <yourname>@cs.uni-freiburg.de
    cd /work/dlclarge2/<yourname>-HPOBENCH

### 2) Clone necessary repositories

    git clone https://github.com/automl/HPOBenchExperimentUtils.git
    cd HPOBenchExperimentUtils
    git checkout developement
    cd ..

    git clone https://github.com/automl/HPOBench.git
    cd HPOBench
    git checkout development
    cd ..

    git clone https://github.com/automl/DEHB.git

### 3) Create Python Env

    conda create -n hpobench_37
    conda install python=3.7

#### Install SMAC
    conda install gxx_linux-64 gcc_linux-64 swig   
    pip install smac[all]

#### Install DEHB
    cd DEHB
    pip install -r requirements.txt 
    cd ..

#### Install HPBandSter
    pip install hpbandster

#### Install Dragonfly
    pip install dragonfly-opt -v

#### Install AutoGluon
    TBD

#### Install HPOBench
    cd HPOBench
    pip install -r requirements.txt
    pip install .

### 4) Edit startup.sh

    nano HPOBenchExperimentUtils/scripts/startup.sh 
    # edit PYTHONPATH to point to correct DEHB location

### 5) Create cmd files, e.g.

First, have a look at this [spreadsheet](https://docs.google.com/spreadsheets/d/1SYFAsL7bm9WhSHwdmdpxK4S7M8ASoAfl4TEA8AtrV4E/edit#gid=1492359258). Then pick an experiment that has not yet been run. There you can also see how much memory your experiment needs. If your experiment is not yet listed, please create a new row/column.

We will first create cmd files and then submit them as batch jobs to the cluster. Please look at `scripts/create_cmd.py`; It is easier to read than to explain, if not ask me.

Then created the files, e.g. 

    EXP=NAS201
    for OPT in rs dehb hpband smac autogluon dragonfly sf
    do
        python create_cmd.py --opt $OPT --nrep 32 --exp $EXP
    done

This will create many \*.cmd files. Each line of each script needs to be run on the cluster.

  * run_* runs the optimizer
  * evalunv_* creates plots on the optimized performance
  * val_* would validate all trajectories and compute the test trajectory (not needed)
  * eval_* creates plots on the test performance (not needed)

### 6) Run experiments

You'll do this with the metahelper. Either try to copy my scripts from `/home/eggenspk/scripts/metahelper/` or clone the repo from [here](https://github.com/automl/helper_scripts)

    python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake --memory 6000 --timelimit 345600 --startup scripts/startup.sh -o ./sgeout -l ./sgeout scripts/CMD/run_NAS201_smac_32.cmd

Tip: Always submit all jobs for one experiment at once. Append `--hold` to put everything in the queue and then release them bit by bit via `scontrol release <jobid>`

Rule of thumbs for surrogate experiments

  * RS, DE, DEHB, HB, HPBAndSter are fast
  * All versions of SMAC are slow
  * Autogluon always takes 4 days (345600 seconds)
  * Dragonfly mostly takes 4 days (345600 seconds)

Rule of thumbs for other experiments: They will take exactly as long as described in [here](https://github.com/automl/HPOBenchExperimentUtils/blob/master/HPOBenchExperimentUtils/benchmark_settings.yaml)

### 7) Some sanity checks

Once a couple of experiments are done, here are some sanity checks

#### Check whether all runs completed

**How**: v1 is always written, v2 only if the run is completed correctly. If there are different numbers of file, figure out which runs does not have *v2*

    ls  exp_outputs/ParamNetReducedOp*/smac_*/run-*/*v2* | wc -l
    ls  exp_outputs/ParamNetReducedOp*/smac_*/run-*/*v1* | wc -l

##### Why 1: mem-outs

    grep oom sgeout/run_paramnettime_smac*

Sometimes 12GB are just not enough. In this case there is nothing you can do. But, please note if this happened.

##### Why 2: The optimizer is just too slow to run within 4 days. In this case check the last line of the runhistory in this folder. 

    tail -n 1 exp_outputs/ParamNetReducedOp*/smac_bo/run-4/hpobench_runhistory.txt 
    head -n 1 exp_outputs/ParamNetReducedOp*/smac_bo/run-4/hpobench_runhistory.txt 

The value of total_time_used tells you how much of the given budget the experiment used (=simulated runtime)
If you subtract boot_time from finish_time you get the actual runtime for that job. In this case there is nothing you can do. But, please note if this happened.

##### Why 3: Other reasons
Also check a few other runs from that optimizer on that benchmark to see whether one run looks suspicious. Please rerun these.

##### Fix 1: If the reasons are unclear, here's what you can do: Remove the directory of that run, e.g. run-23, and run the whole batch job again. The experiments are written s.t. only jobs for which the output file does not exist run.

##### Fix 2: If there reasons are slow optimizers or memouts, here's what you can do:

    python HPOBenchExperimentUtils/extract_trajectory.py --output_dir exp_outputs/ParamNetReducedMnist*/dragonfly_*/ --debug

This will create all v2 trajectories, but please only do that if you are really sure that rerunning won't fix the issue

### 8) Plotting

Same here, please use your own version of the helper script. Here, it is *okay* to submit to the testqueue since the jobs are fast, but require a lot of memory.

    python /home/eggenspk/scripts/metahelper/slurm_helper.py -q testbosch_cpu-cascadelake --memory 32000 --timelimit 3600 --startup scripts/startup.sh -o ./sgeout -l ./sgeout scripts/CMD/evalunv_NAS201_*.cmd

To get the *.tex table:

    for i in plots_tab/result_table_*_v1.tex; do head -n 5 $i | tail -n 1; done

Also at least once open the *stats* file for each experiment and scan it for suspicious outliers.

### 9) Finally, mark the experiment as completed and continue
