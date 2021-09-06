source ~/.bashrc
source $HOME/allow_internet.sh

source ~/anaconda3/bin/activate
conda activate hpobench_37

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export PYTHONPATH=/work/dlclarge1/mallik-hpobench/DEHB:$PYTHONPATH

# export HTTP_PROXY=http://tfproxy.informatik.uni-freiburg.de:8080
# export HTTPS_PROXY=http://tfproxy.informatik.uni-freiburg.de:8080
# git config --global http.proxy $HTTP_PROXY
# git config --global https.proxy $HTTPS_PROXY
# export http_proxy=http://tfproxy.informatik.uni-freiburg.de:8080
# export https_proxy=http://tfproxy.informatik.uni-freiburg.de:8080
