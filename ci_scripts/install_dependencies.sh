#!/usr/bin/env sh

install_packages=""

echo "Install tools for ${OPTIMIZER}"

# Install HPOBench
git clone https://github.com/automl/HPOBench.git
cd HPOBench
git checkout AddStartupAddConnectClient  # TODO: PM after merge in the HPOBench use the dev branch.
pip install "."
cd ..

if [[ "$OPTIMIZER" == "autogluon" ]];
then
    install_packages="${install_packages}autogluon,"
    pip install --upgrade setuptools pip

elif [[ "$OPTIMIZER" == "dehb" ]];
then
    install_packages="${install_packages}dehb,"
    git clone https://github.com/automl/DEHB.git
    cd DEHB
    git checkout b8dcba7b38bf6e7fc8ce3e84ea567b66132e0eb5
    cd ..
    export PYTHONPATH=~/DEHB:$PYTHONPATH
    export PYTHONPATH=$PWD/DEHB:$PYTHONPATH
    echo $PYTHONPATH

# elif [[ "$OPTIMIZER" == "dragonfly_default" ]];
# then

elif [[ "$OPTIMIZER" == "hpbandster_bohb_eta_3" ]];
then
    install_packages="${install_packages}hpbandster,"

elif [[ "$OPTIMIZER" == "ray_hyperopt_hb" ]];
then
    install_packages="${install_packages}ray_base,ray_hyperopt,"

elif [[ "$OPTIMIZER" == "ray_bayesopt_hb" ]];
then
    install_packages="${install_packages}ray_base,ray_bayesopt,"

elif [[ "$OPTIMIZER" == "ray_optuna_hb" ]];
then
    install_packages="${install_packages}ray_base,ray_optuna,"

elif [[ "$OPTIMIZER" == "smac_hb_eta_3" ]];
then
    install_packages="${install_packages}smac,"

else
    echo "NO PACKAGE WAS INSTALLED! THAT IS NOT GOOD!"
    exit 1
fi

# remove the trailing comma
install_packages="$(echo ${install_packages} | sed 's/,*\r*$//')"
echo "Install EXPUtils with options: ${install_packages}"
pip install .["${install_packages}"]
