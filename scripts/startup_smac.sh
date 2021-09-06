source ~/.bashrc
source $HOME/allow_internet.sh

source ~/anaconda3/bin/activate
conda activate hpobench_37_smac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export PYTHONPATH=/work/dlclarge1/mallik-hpobench/DEHB:$PYTHONPATH
