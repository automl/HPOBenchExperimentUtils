from pathlib import Path
from HPOBenchExperimentUtils.run_benchmark import run_benchmark

benchmark = 'ParamNetPokerOnTimeBenchmark'
output_dir = Path('./DEBUG_Autogluon/')
rng = 1

# optimizer = 'randomsearch'
optimizer = 'autogluon'
# optimizer = 'smac_hb_eta_3'

res_folder = output_dir / f'{benchmark}/{optimizer}/run-{rng}'

if res_folder.exists():
    import shutil
    shutil.rmtree(res_folder)

run_benchmark(optimizer=optimizer,
              benchmark=benchmark,
              output_dir=output_dir,
              rng=rng,
              use_local=False)

print('finished_run')
