from pathlib import Path
from HPOBenchExperimentUtils.run_benchmark import run_benchmark

benchmark = 'ParamNetPokerOnTimeBenchmark'
output_dir = Path('./NASCifar10ABenchmark')
rng = 1

# optimizer = 'randomsearch'
optimizer = 'autogluon'
res_folder = Path(f'./DEBUG_Autogluon/ParamNetPokerOnTimeBenchmark/{optimizer}/run-1')

if res_folder.exists():
    import shutil
    shutil.rmtree(res_folder)

run_benchmark(optimizer=optimizer,
              benchmark=benchmark,
              output_dir=output_dir,
              rng=rng,
              use_local=False)

print('finished_run')
