import submitit
from datetime import datetime
import numpy as np
import argparse
import main_batch
import json
import os

parser = argparse.ArgumentParser()
def create_shared_parser(**kwargs):
    parser = argparse.ArgumentParser()
    for k, v in kwargs.items():
        new_k = k.replace('_', '-')
        
        if isinstance(v, list) and len(v) > 0:
            # Assume all elements are of the same type as the first element
            elem_type = type(v[0])
            parser.add_argument(f"--{new_k}", type=elem_type, nargs='+', default=v)
        
        elif isinstance(v, bool):
            if v:
                parser.add_argument(f"--{new_k}", action="store_true")
            else:
                parser.add_argument(f"--{new_k}", action="store_false")
        
        else:
            parser.add_argument(f"--{new_k}", type=type(v), default=v)

    return parser


experiment_name = 'n_sweep_optimal_lr'

Ts= [13]
ns=[i for i in range(5,14)]  # student hidden units
lrs=[1.0]
r_1s=[1.0]
r_2s=[0.0]
epsilons=[0.0]
sigma_1s=[1.0]
sigma_2s=[1.0]
baseline=False
optimal_LR_bool=True
D = 400
epochs=10000
savepoints=100

deltas = [0.0]
rhos=[0.0]



base_folder_name = f'T{Ts[0]}lr{lrs[0]}r_1{r_1s[0]}r_2{r_2s[0]}rho{rhos[0]}delta{deltas[0]}sigma1{sigma_1s[0]}sigma2{sigma_2s[0]}epsilon{epsilons[0]}baseline{baseline}epochs{epochs}savepoints{savepoints}'

"""
change actual_folder_name to reflect the sweep being run
"""
actual_folder_name = f'T{Ts[0]}lr{lrs[0]}r_1{r_1s[0]}r_2{r_2s[0]}rho{rhos[0]}delta{deltas[0]}sigma1{sigma_1s[0]}sigma2{sigma_2s[0]}epsilon{epsilons[0]}baseline{baseline}epochs{epochs}savepoints{savepoints}'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
basepath = f'/ceph/scratch/npatel/le_cake/single_sweep_results/{experiment_name}/{timestamp}/{actual_folder_name}'
log_path = f'./logs/{timestamp}/'
run_name = 'ode_sweep'

if not os.path.isdir(basepath):
    os.makedirs(basepath)

shared_args = create_shared_parser(
    Ts=Ts,
    ns=ns,
    lrs=lrs,
    r_1s=r_1s,
    r_2s=r_2s,
    deltas=deltas,
    rhos=rhos,
    epsilons=epsilons,
    sigma_1s=sigma_1s,
    sigma_2s=sigma_2s,
    baseline=not baseline,
    optimal_LR=not optimal_LR_bool,
    logdir=basepath,
    D=D,
    epochs=epochs,
    savepoints=savepoints
).parse_args()

config_dict = {'fixed':{}, 'sweep':{}}
for k,v in vars(shared_args).items():
    if type(v) == list and len(v) ==1:
        config_dict['fixed'][k] = v
    elif not type(v) == list:
        print(v, type(v))
        config_dict['fixed'][k] = v
    else:
        config_dict['sweep'][k] = v

json.dump(config_dict, open(f'{basepath}/config.json', 'w'), indent=4)

all_args = []
all_args.append(shared_args)

print('fucking go on then')

os.makedirs(log_path, exist_ok=True)
executor = submitit.AutoExecutor(folder=f"{log_path}")

executor.update_parameters(name=run_name,
                           nodes=1,
                           gpus_per_node=1,
                           tasks_per_node=1,
                           cpus_per_task=4,
                           timeout_min=60,
                           slurm_partition='gpu_saxe')

#jobs = executor.map_array(main_batch.main_batched, all_args)
jobs = executor.map_array(main_batch.main_batched_N_more, all_args)
