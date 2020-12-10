source ~/.bashrc

conda activate hpobench_37
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export PYTHONPATH=~/2020_Hpolib2/DEHB:$PYTHONPATH
