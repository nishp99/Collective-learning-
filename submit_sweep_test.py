import submitit
from datetime import datetime
import numpy as np
import argparse
import main_ode
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



Ts= [6]
lrs=[1.0]
r_1s=[1.0]
r_2s=[1.0]
r_12s=[1.0]
tau_1s=[0.0]
tau_2s=[0.0]
D = 1000
epochs=1000
savepoints=100

"""
can more specifically control the initial values of the order parameters here if desired
"""
J_1_inits=[0 + np.random.normal(0, 1/np.sqrt(D))]
J_2_inits=[0 + np.random.normal(0, 1/np.sqrt(D))]
Q_1_inits=[1 + np.random.normal(0, 1/np.sqrt(D))]
Q_2_inits=[1 + np.random.normal(0, 1/np.sqrt(D))]
Q_12_inits=[0 + np.random.normal(0, 1/np.sqrt(D))]
S_s = [1 + np.random.normal(0, 1/np.sqrt(D))]


                        #is the file name structure below fine for you?
base_folder_name = f'T{Ts[0]}lr{lrs[0]}r_1{r_1s[0]}r_2{r_2s[0]}r_12{r_12s[0]}tau_1{tau_1s[0]}tau_2{tau_2s[0]}J_1{J_1_inits[0]}J_2{J_2_inits[0]}Q_1{Q_1_inits[0]}Q_2{Q_2_inits[0]}Q_12{Q_12_inits[0]}S{S_s[0]}epochs{epochs}savepoints{savepoints}'

"""
change actual_folder_name and experiment name to reflect the sweep being run
"""
experiment_name = 'test_ode'
actual_folder_name = f'T{Ts[0]}lr{lrs[0]}r_1{r_1s[0]}r_2{r_2s[0]}r_12{r_12s[0]}tau_1{tau_1s[0]}tau_2{tau_2s[0]}J_1{J_1_inits[0]}J_2{J_2_inits[0]}Q_1{Q_1_inits[0]}Q_2{Q_2_inits[0]}Q_12{Q_12_inits[0]}S{S_s[0]}epochs{epochs}savepoints{savepoints}'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
basepath = f'/ceph/scratch/npatel/collective_learning/{experiment_name}/{timestamp}/{actual_folder_name}'
log_path = f'./logs/{timestamp}/'
run_name = 'ode_la_bomba_pow'

if not os.path.isdir(basepath):
    os.makedirs(basepath)

shared_args = create_shared_parser(
    Ts=Ts,
    lrs=lrs,
    r_1s=r_1s,
    r_2s=r_2s,
    r_12s=r_12s,
    tau_1s=tau_1s,
    tau_2s=tau_2s,
    J_1_inits=J_1_inits,
    J_2_inits=J_2_inits,
    Q_1_inits=Q_1_inits,
    Q_2_inits=Q_2_inits,
    Q_12_inits=Q_12_inits,
    S_s=S_s,
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

print('Submitting jobs')

os.makedirs(log_path, exist_ok=True)
executor = submitit.AutoExecutor(folder=f"{log_path}")

executor.update_parameters(name=run_name,
                           nodes=1,
                           gpus_per_node=1,
                           tasks_per_node=1,
                           cpus_per_task=8,
                           timeout_min=20,
                           slurm_partition='gpu_saxe')

jobs = executor.map_array(main_ode.main_batched, all_args)
