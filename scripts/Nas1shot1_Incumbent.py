import sys
sys.path.append('~/nasbench-1shot1')

import numpy as np
from tqdm import tqdm
from itertools import product
from ast import literal_eval
from hpobench.benchmarks.nas.nasbench_1shot1 import NASBench1shot1SearchSpace1Benchmark,\
    NASBench1shot1SearchSpace2Benchmark, NASBench1shot1SearchSpace3Benchmark


def compute_incumbents(benchmark):
    cs = benchmark.get_configuration_space()

    hps = cs.get_hyperparameters()
    hps_values = [hp.choices for hp in hps]
    hps_names = [hp.name for hp in hps]

    # build all possible configurations
    all_combinations = list(product(*hps_values))

    inc_train_c, inc_train_v = {}, 0
    inc_valid_c, inc_valid_v = {}, 0
    inc_test_c,  inc_test_v  = {}, 0

    avg_inc_train_c, avg_inc_train_v = {}, 0
    avg_inc_valid_c, avg_inc_valid_v = {}, 0
    avg_inc_test_c,  avg_inc_test_v  = {}, 0

    print(f'Number of Configurations: {len(all_combinations)}')

    for config in tqdm(all_combinations):
        config = {name: value for name, value in zip(hps_names, config)}
        config = parse_configuration(config)

        result_seed_0 = benchmark._query_benchmark(config, fidelity={'budget': 108}, run_index=0)
        result_seed_1 = benchmark._query_benchmark(config, fidelity={'budget': 108}, run_index=1)
        result_seed_2 = benchmark._query_benchmark(config, fidelity={'budget': 108}, run_index=2)

        # Calculate the incumbent for all configuraions and neglecting the seed
        inc_train_c, inc_train_v = set_incumbent(inc_train_c, inc_train_v, config,
                                                 result_seed_0, result_seed_1, result_seed_2, 'train_accuracy')

        inc_valid_c, inc_valid_v = set_incumbent(inc_valid_c, inc_valid_v, config,
                                                 result_seed_0, result_seed_1, result_seed_2, 'validation_accuracy')

        inc_test_c, inc_test_v = set_incumbent(inc_test_c, inc_test_v, config,
                                               result_seed_0, result_seed_1, result_seed_2, 'test_accuracy')

        avg_inc_train_c, avg_inc_train_v = set_incumbent_avg(avg_inc_train_c, avg_inc_train_v, config,
                                                             result_seed_0, result_seed_1, result_seed_2,
                                                             'train_accuracy')

        avg_inc_valid_c, avg_inc_valid_v = set_incumbent_avg(avg_inc_valid_c, avg_inc_valid_v, config,
                                                             result_seed_0, result_seed_1, result_seed_2,
                                                             'validation_accuracy')

        avg_inc_test_c, avg_inc_test_v = set_incumbent_avg(avg_inc_test_c, avg_inc_test_v, config,
                                                           result_seed_0, result_seed_1, result_seed_2,
                                                           'test_accuracy')

    print(f'SINGLE SEED: \n'
          f'Train: {inc_train_v}'
          f'Valid: {inc_valid_v}'
          f'Test: {inc_test_v}')

    print(f'AVG SEED: \n'
          f'Train: {avg_inc_train_v}'
          f'Valid: {avg_inc_valid_v}'
          f'Test: {avg_inc_test_v}')

    return inc_train_c, inc_train_v, inc_valid_c, inc_valid_v, inc_test_c,  inc_test_v, \
        avg_inc_train_c, avg_inc_train_v, avg_inc_valid_c, avg_inc_valid_v, avg_inc_test_c,  avg_inc_test_v


def parse_configuration(configuration):
    # make sure that it is a dictionary and not a CS.Configuration.
    return {k: literal_eval(v) if isinstance(v, str) and v[0] == '(' else v
            for k, v in configuration.items()}


def set_incumbent(inc_c, inc_v, config, s0, s1, s2, metric):
    for s in [s0, s1, s2]:
        if inc_v < s[metric]:
            inc_c, inc_v = config, s[metric]

    return inc_c, inc_v


def set_incumbent_avg(inc_c, inc_v, config, s0, s1, s2, metric):
    avg_accuracy = np.mean([s0[metric], s1[metric], s2[metric]])

    if inc_v < avg_accuracy:
        inc_c, inc_v = config, avg_accuracy

    return inc_c, inc_v


results_space_1 = compute_incumbents(NASBench1shot1SearchSpace1Benchmark())
results_space_2 = compute_incumbents(NASBench1shot1SearchSpace2Benchmark())
results_space_3 = compute_incumbents(NASBench1shot1SearchSpace3Benchmark())

def print_result(result):
    from pprint import pprint
    inc_train_c, inc_train_v, inc_valid_c, inc_valid_v, inc_test_c, inc_test_v, \
        avg_inc_train_c, avg_inc_train_v, avg_inc_valid_c, avg_inc_valid_v, avg_inc_test_c, avg_inc_test_v = result

    print(f'SINGLE SEED: \n'
          f'Train: {inc_train_v}\n'
          f'Valid: {inc_valid_v}\n'
          f'Test: {inc_test_v}\n')

    print(f'AVG SEED: \n'
          f'Train: {avg_inc_train_v}\n'
          f'Valid: {avg_inc_valid_v}\n'
          f'Test: {avg_inc_test_v}\n')

    print('Inc Config Train')
    pprint(inc_train_c)

    print('Inc Config Valid')
    pprint(inc_valid_c)

    print('Inc Config Test')
    pprint(inc_test_c)

    print('AVG: Inc Config Train')
    pprint(avg_inc_train_c)

    print('AVG: Inc Config Valid')
    pprint(avg_inc_valid_c)

    print('AVG: Inc Config Test')
    pprint(avg_inc_test_c)

print(80*'#')
print('RESULT SPACE 1\n')
print_result(results_space_1)

print(80*'#')
print('RESULT SPACE 2\n')
print_result(results_space_2)

print(80*'#')
print('RESULT SPACE 3\n')
print_result(results_space_3)


"""
Max across Seeds
----------------
        Search Space 1          Search Space 2          Search Space 3
Train   1.0                     1.0                     1.0
Valid   0.9507211446762085      0.9484174847602844      0.9515224099159241
Test    0.9455128312110901      0.9424078464508057      0.9466145634651184


Incumbent averaged across Seeds
-------------------------------
        Search Space 1          Search Space 2          Search Space 3
Train   1.0                     1.0                     1.0
Valid   0.9471821784973145      0.9456797440846761      0.9473824898401896
Test    0.9420072237650553      0.9396701256434122      0.941773513952891


Number of Configurations (with each 3 Seeds)
--------------------------------------------
        Search Space 1          Search Space 2          Search Space 3
        14580                   29160                   1312200


Output of the script with the best found configurations:
--------------------------------------------------------

################################################################################
RESULT SPACE 1

SINGLE SEED: 
Train: 1.0
Valid: 0.9507211446762085
Test: 0.9455128312110901

AVG SEED: 
Train: 1.0
Valid: 0.9471821784973145
Test: 0.9420072237650553

Inc Config Train
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0, 1),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 1),
 'choice_block_5_parents': (0, 4)}
Inc Config Valid
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0, 1),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (2, 3),
 'choice_block_5_parents': (0, 4)}
Inc Config Test
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0, 1),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (1, 2),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_parents': (0, 4)}
AVG: Inc Config Train
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0, 1),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 1),
 'choice_block_5_parents': (0, 4)}
AVG: Inc Config Valid
{'choice_block_1_op': 'maxpool3x3',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0, 1),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (0, 2),
 'choice_block_4_op': 'conv1x1-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_parents': (0, 4)}
AVG: Inc Config Test
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0, 1),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (1, 2),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_parents': (0, 4)}
################################################################################
RESULT SPACE 2

SINGLE SEED: 
Train: 1.0
Valid: 0.9484174847602844
Test: 0.9424078464508057

AVG SEED: 
Train: 1.0
Valid: 0.9456797440846761
Test: 0.9396701256434122

Inc Config Train
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 1),
 'choice_block_5_parents': (0, 1, 4)}
Inc Config Valid
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (1,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (0, 2),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_parents': (0, 1, 4)}
Inc Config Test
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (0, 2),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_parents': (0, 1, 4)}
AVG: Inc Config Train
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 1),
 'choice_block_5_parents': (0, 1, 4)}
AVG: Inc Config Valid
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 2),
 'choice_block_5_parents': (0, 3, 4)}
AVG: Inc Config Test
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (0, 1),
 'choice_block_4_op': 'conv1x1-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_parents': (0, 2, 4)}
################################################################################
RESULT SPACE 3

SINGLE SEED: 
Train: 1.0
Valid: 0.9515224099159241
Test: 0.9466145634651184

AVG SEED: 
Train: 1.0
Valid: 0.9473824898401896
Test: 0.941773513952891

Inc Config Train
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (0,),
 'choice_block_4_op': 'conv1x1-bn-relu',
 'choice_block_4_parents': (0, 1),
 'choice_block_5_op': 'conv3x3-bn-relu',
 'choice_block_5_parents': (0, 1),
 'choice_block_6_parents': (0, 5)}
Inc Config Valid
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (1,),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_op': 'conv1x1-bn-relu',
 'choice_block_5_parents': (2, 4),
 'choice_block_6_parents': (0, 5)}
Inc Config Test
{'choice_block_1_op': 'conv3x3-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (1,),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_op': 'conv1x1-bn-relu',
 'choice_block_5_parents': (2, 4),
 'choice_block_6_parents': (0, 5)}
AVG: Inc Config Train
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv1x1-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'conv1x1-bn-relu',
 'choice_block_3_parents': (0,),
 'choice_block_4_op': 'conv1x1-bn-relu',
 'choice_block_4_parents': (0, 1),
 'choice_block_5_op': 'conv3x3-bn-relu',
 'choice_block_5_parents': (0, 1),
 'choice_block_6_parents': (0, 5)}
AVG: Inc Config Valid
{'choice_block_1_op': 'conv1x1-bn-relu',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (0,),
 'choice_block_3_op': 'maxpool3x3',
 'choice_block_3_parents': (2,),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (0, 3),
 'choice_block_5_op': 'conv1x1-bn-relu',
 'choice_block_5_parents': (0, 4),
 'choice_block_6_parents': (0, 5)}
AVG: Inc Config Test
{'choice_block_1_op': 'maxpool3x3',
 'choice_block_1_parents': (0,),
 'choice_block_2_op': 'conv3x3-bn-relu',
 'choice_block_2_parents': (1,),
 'choice_block_3_op': 'conv3x3-bn-relu',
 'choice_block_3_parents': (2,),
 'choice_block_4_op': 'conv3x3-bn-relu',
 'choice_block_4_parents': (2, 3),
 'choice_block_5_op': 'conv3x3-bn-relu',
 'choice_block_5_parents': (0, 4),
 'choice_block_6_parents': (0, 5)}

"""