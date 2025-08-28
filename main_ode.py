import numpy as np
import multiprocessing
import argparse
import os
#import joblib as jl

def configure_jax_cpu_threads():
    # Detect CPU thread count from SLURM or fallback
    cpus_per_task = (
        int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or
        int(os.environ.get("OMP_NUM_THREADS", 0)) or
        os.cpu_count() or
        1
    )
    print('SLURM_CPUS_PER_TASK:', os.environ.get("SLURM_CPUS_PER_TASK", 0))
    print('OMP_NUM_THREADS:', os.environ.get("OMP_NUM_THREADS", 0))
    print('os.cpu_count():', os.cpu_count())

    # Set XLA and threading environment variables
    os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true"
    os.environ["OMP_NUM_THREADS"] = str(cpus_per_task)
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

    print(f"[JAX CONFIG] Using {cpus_per_task} CPU threads")

configure_jax_cpu_threads()

import jax.numpy as jnp
from jax import random, jit, lax, vmap
import jax
from jax.lib import xla_bridge

import itertools
import pandas as pd
import time
from datetime import datetime
import json
"""
ODE functions
"""

def order_params_ode(student_1, student_2, teacher, D):
    J_1 = jnp.einsum('D, D -> ', student_1, teacher) / D
    J_2 = jnp.einsum('D, D -> ', student_2, teacher) / D
    Q_1 = jnp.einsum('D, D -> ', student_1, student_1) / D
    Q_2 = jnp.einsum('D, D -> ', student_2, student_2) / D
    Q_12 = jnp.einsum('D, D -> ', student_1, student_2) / D
    return J_1, J_2, Q_1, Q_2, Q_12

"""
functions to calculate expectations of relevant aligning field quantities
"""
def ModNu(S):
    return jnp.sqrt(2 * S/ jnp.pi)

def ModLambda(Q):
    return jnp.sqrt(2 * Q / jnp.pi)

def SgnLambda_Nu(J, S):
    return jnp.sqrt(2 / jnp.pi) * J / jnp.sqrt(S)

def SgnLambdai_Lambdaj(Q_i, Q_ij):
    return jnp.sqrt(2 / jnp.pi) * Q_ij / jnp.sqrt(Q_i)

def SgnNu_Lambda(J, Q):
    return jnp.sqrt(2 / jnp.pi) * J / jnp.sqrt(Q)

def Sgn12(Q_1, Q_2, Q_12):
    return 2 / jnp.pi * jnp.arcsin(Q_12 / jnp.sqrt(Q_1 * Q_2))

def SgnNuLambda(J, Q, S):
    return 2 / jnp.pi * jnp.arcsin(J / jnp.sqrt(Q * S))

def Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (Q_12 * S - J_1 * J_2) / jnp.sqrt((Q_1 * S - J_1**2) * (Q_2 * S - J_2**2))

def Rho_lambda(J_i, J_j, Q_i, Q_j, Q_12, S):
    return (J_i * Q_j - J_j * Q_12) / jnp.sqrt((Q_i * S - J_i**2) * (Q_j * S - J_j**2))

def HeaviLambdaNu(J, Q):
    return 1 - 1 / jnp.pi * jnp.arccos(J / Q)
    
def Heavi12_heavi12Nu(J_1, J_2, Q_1, Q_2, Q_12, S):
    prob = 1/4 + 1/(2 * jnp.pi) * (jnp.arcsin(Q_12 / jnp.sqrt(Q_1*Q_2)) + jnp.arcsin(J_1/jnp.sqrt(Q_1*S)) + jnp.arcsin(J_2/jnp.sqrt(Q_2*S)))
    return prob

def Heavi12_negheavi12Nu(J_1, J_2, Q_1, Q_2, Q_12, S):
    prob = 1/(2 * jnp.pi) * (-jnp.arccos(Q_12 / jnp.sqrt(Q_1*Q_2)) + jnp.arccos(J_1/jnp.sqrt(Q_1*S)) + jnp.arccos(J_2/jnp.sqrt(Q_2*S)))
    return prob

def Sgn12_modNu(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (jnp.sqrt(2 / jnp.pi) * (J_1/jnp.sqrt(Q_1) + J_2/jnp.sqrt(Q_2) + jnp.sqrt(S)) - 
            4/(jnp.pi * jnp.sqrt(2*jnp.pi)) * (J_1/jnp.sqrt(Q_1) * jnp.arccos(Rho_lambda(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        J_2/jnp.sqrt(Q_2) * jnp.arccos(Rho_lambda(J_2, J_1, Q_2, Q_1, Q_12, S)) +
                        jnp.sqrt(S) * jnp.arccos(Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S))))

def SgniNu_modj(J_i, J_j, Q_i, Q_j, Q_12, S):
    return 2 * jnp.sqrt(2) / (jnp.pi * jnp.sqrt(jnp.pi)) * (jnp.sqrt(Q_j) * jnp.arcsin(Rho_lambda(J_j, J_i, Q_j, Q_i, Q_12, S)) +
                        Q_12 / jnp.sqrt(Q_i) * jnp.arcsin(Rho_lambda(J_i, J_j, Q_i, Q_j, Q_12, S)) +
                        J_j / jnp.sqrt(S) * jnp.arcsin(Rho_12(J_i, J_j, Q_i, Q_j, Q_12, S)))

    
"""
calculate order parameter updates. YO NISH, I changed a sign somewhere, good luck finding it :(
"""
def dJ_i(T, eta, r_i, r_j, r_12, tau_j, sgni_Nu, sgnj_Nu, sgn12_modNu, modNu, P_i, P_j, P_12_collab):
    dj = eta * (r_i / 2 * P_i**(T - 1) * (sgni_Nu + modNu) + tau_j * r_j / 2 * P_j**(T - 1) * (sgn12_modNu + sgni_Nu)
                  + r_12 / 4 * P_12_collab**(T - 1) * (sgni_Nu + sgnj_Nu - sgn12_modNu - modNu))
    return dj

def dQ_i(T, eta, r_i, r_j, r_12, tau_j, modi, modj, sgnNu_i, sgnjNu_modi, sgnj_i, 
         P_i, P_j, P_12, P_12_collab):
    dq = (eta * (r_i * P_i**(T - 1) * (modi + sgnNu_i) + tau_j * r_j * P_j**(T - 1) * (modi + sgnjNu_modi) + 
                 r_12 / 2 * P_12_collab**(T - 1) * (modi + sgnj_i - sgnjNu_modi - sgnNu_i)) + 
                 eta**2 / T * (r_i**2 * P_i**T + tau_j * r_j**2 * P_j**T + 
                               r_12**2 * P_12_collab**T + r_i * r_j * tau_j * P_12**T))
    return dq

def dQ_ij(T, eta, r_1, r_2, r_12, tau_1, tau_2, mod1, mod2, sgn1_2, sgn2_1, sgnNu_1, 
          sgnNu_2, sgn12, sgnNu1, sgnNu2, sgn1Nu_mod2, sgn2Nu_mod1, P_1, P_2, P_12, P_12_collab):
    dq_ij = (eta / 2 * (r_1 * (sgn1_2 + sgnNu_2 + tau_1 * (sgn2_1 + sgn2Nu_mod1)) * P_1**(T - 1) + r_2 * (sgn2_1 + sgnNu_1 + tau_2 * (sgn1_2 + sgn1Nu_mod2)) * P_2**(T - 1) + 
                    r_12 / 4 * (mod2 + sgn1_2 - sgn1Nu_mod2 - sgnNu_2 + mod1 + sgn2_1 - sgn2Nu_mod1 - sgnNu_1) * P_12_collab**(T - 1)) + 
                    eta**2 / T * ((1 + tau_1 * tau_2) * r_1 * r_2 / 4 * (1 + sgn12 + sgnNu1 + sgnNu2) * P_12**(T - 1) + 
                                  r_12**2 / 4 * (1 + sgn12 - sgnNu1 - sgnNu2) * P_12_collab**(T - 1) + 
                                  tau_1 * r_1**2 / 2 * (sgn12 + sgnNu2) * P_1**(T - 1) + tau_2 * r_2**2 / 2 * (sgn12 + sgnNu1) * P_2**(T - 1)))
    return dq_ij


"""
update D times for the ODE solver
"""
def update_D_times(dt, eta, T, r_1, r_2, r_12, tau_1, tau_2, J_1, J_2, Q_1, Q_2, Q_12, S):
    D = 400
    def body(_, state):
        J_1, J_2, Q_1, Q_2, Q_12 = state

        P_1 = HeaviLambdaNu(J_1, Q_1)
        P_2 = HeaviLambdaNu(J_2, Q_2)
        P_12 = Heavi12_heavi12Nu(J_1, J_2, Q_1, Q_2, Q_12, S)
        P_12_collab = Heavi12_negheavi12Nu(J_1, J_2, Q_1, Q_2, Q_12, S)

        sgn12_modNu = Sgn12_modNu(J_1, J_2, Q_1, Q_2, Q_12, S)
        sgn1Nu_mod2 = SgniNu_modj(J_1, J_2, Q_1, Q_2, Q_12, S)
        sgn2Nu_mod1 = SgniNu_modj(J_2, J_1, Q_2, Q_1, Q_12, S)

        sgn1_Nu = SgnLambda_Nu(J_1, S)
        sgn2_Nu = SgnLambda_Nu(J_2, S)
        sgnNu_1 = SgnNu_Lambda(J_1, Q_1)
        sgnNu_2 = SgnNu_Lambda(J_2, Q_2)
        sgn1_2 = SgnLambdai_Lambdaj(Q_1, Q_12)
        sgn2_1 = SgnLambdai_Lambdaj(Q_2, Q_12)

        sgn12 = Sgn12(Q_1, Q_2, Q_12)
        sgnNu1 = SgnNuLambda(J_1, Q_1, S)
        sgnNu2 = SgnNuLambda(J_2, Q_2, S)

        mod1 = ModLambda(Q_1)
        mod2 = ModLambda(Q_2)
        modNu = ModNu(S)
               
        dJ_1 = dJ_i(T, eta, r_1, r_2, r_12, tau_2, sgn1_Nu, sgn2_Nu, sgn12_modNu, modNu, P_1, P_2, P_12_collab) * dt
        dJ_2 = dJ_i(T, eta, r_2, r_1, r_12, tau_1, sgn2_Nu, sgn1_Nu, sgn12_modNu, modNu, P_2, P_1, P_12_collab) * dt
        dQ_1 = dQ_i(T, eta, r_1, r_2, r_12, tau_2, mod1, mod2, sgnNu_1, sgn2Nu_mod1, sgn2_1, P_1, P_2, P_12, P_12_collab) * dt
        dQ_2 = dQ_i(T, eta, r_2, r_1, r_12, tau_1, mod2, mod1, sgnNu_2, sgn1Nu_mod2, sgn1_2, P_2, P_1, P_12, P_12_collab) * dt
        dQ_12 = dQ_ij(T, eta, r_1, r_2, r_12, tau_1, tau_2, mod1, mod2, sgn1_2, sgn2_1, sgnNu_1, sgnNu_2, sgn12, sgnNu1, 
                      sgnNu2, sgn1Nu_mod2, sgn2Nu_mod1, P_1, P_2, P_12, P_12_collab) * dt

        J_1 = J_1 + dJ_1
        J_2 = J_2 + dJ_2
        Q_1 = Q_1 + dQ_1
        Q_2 = Q_2 + dQ_2
        Q_12 = Q_12 + dQ_12

        return J_1, J_2, Q_1, Q_2, Q_12

    return lax.fori_loop(0, D, body, (J_1, J_2, Q_1, Q_2, Q_12))

"""
jittable way for finding corresponding index
"""
def lookup_index(x, keys, values):
    match = keys == x + 1
    exists = jnp.any(match)              # bool[]
    first_idx = jnp.argmax(match)        # int[]
    return jnp.where(exists, values[first_idx], -1)

"""
numerically solve the ODE for given parameters for given number of epochs,
storing savepoints number of points logarithmically spaced
"""
def ODE_solver(eta, T, r_1, r_2, r_12, tau_1, tau_2, dt, epochs, savepoints, 
               J_1_init=1e-4, J_2_init=1e-4, Q_1_init=1, Q_2_init=1, Q_12_init=1e-4, S=1):
    
    J_1 = J_1_init
    J_2 = J_2_init
    Q_1 = Q_1_init
    Q_2 = Q_2_init
    Q_12 = Q_12_init
    S = S

    log_keys = np.unique(np.logspace(0, np.log10(epochs - 1), num=savepoints, dtype=int))
    save_number = len(log_keys)
    log_keys = jnp.array(log_keys)
    log_values = jnp.arange(save_number, dtype=int)

    def body(e, carry):
        J_1_s, J_2_s, Q_1_s, Q_2_s, Q_12_s, J_1, J_2, Q_1, Q_2, Q_12 = carry

        index = lookup_index(e, log_keys, log_values)
        J_1_s = J_1_s.at[index].set(J_1)
        J_2_s = J_2_s.at[index].set(J_2)
        Q_1_s = Q_1_s.at[index].set(Q_1)
        Q_2_s = Q_2_s.at[index].set(Q_2)
        Q_12_s = Q_12_s.at[index].set(Q_12)

        J_1, J_2, Q_1, Q_2, Q_12 = update_D_times(dt, eta, T, r_1, r_2, r_12, tau_1, tau_2, J_1, J_2, Q_1, Q_2, Q_12, S)

        return J_1_s, J_2_s, Q_1_s, Q_2_s, Q_12_s, J_1, J_2, Q_1, Q_2, Q_12

    #create logspaced index array
    #indices = jnp.logspace(0, jnp.log10(time_steps), num=100, dtype=int)
    J_1_s = jnp.zeros(save_number+1, dtype=jnp.float32)
    J_2_s = jnp.zeros(save_number+1, dtype=jnp.float32)
    Q_1_s = jnp.zeros(save_number+1, dtype=jnp.float32)
    Q_2_s = jnp.zeros(save_number+1, dtype=jnp.float32)
    Q_12_s = jnp.zeros(save_number+1, dtype=jnp.float32)

    J_1_s, J_2_s, Q_1_s, Q_2_s, Q_12_s, _, _, _, _, _ = lax.fori_loop(
        0, epochs,
        body,
        (J_1_s, J_2_s, Q_1_s, Q_2_s, Q_12_s, J_1, J_2, Q_1, Q_2, Q_12)
    )

    #return the storing arrays
    return J_1_s[:-1], J_2_s[:-1], Q_1_s[:-1], Q_2_s[:-1], Q_12_s[:-1]
ODE_solver = jit(ODE_solver, static_argnums=(7, 8, 9))
vmapped_ODE_solver = jax.vmap(ODE_solver, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None))


def main_batched(args):
    jax.config.update("jax_platform_name", "cpu")
    dt = 1 / args.D
    os.makedirs(args.logdir, exist_ok=True)

    #create batch of parameter combinations
    param_lists = [args.lrs, args.Ts, args.r_1s, args.r_2s, args.r_12s, args.tau_1s, args.tau_2s]
    param_combinations = list(itertools.product(*param_lists))
    param_array = jnp.array(param_combinations)
    lr_batch, T_batch, r_1_batch, r_2_batch, r_12_batch, tau_1_batch, tau_2_batch = param_array.T

    #run batch of ODEs
    start = time.time()
    results = vmapped_ODE_solver(lr_batch, T_batch, r_1_batch, r_2_batch, r_12_batch, tau_1_batch, tau_2_batch, dt, args.epochs, args.savepoints,
                                 args.J_1_init, args.J_2_init, args.Q_1_init, args.Q_2_init, args.Q_12_init, args.S)
    
    #store results in pandas dataframe
    J_1_vals, J_2_vals, Q_1_vals, Q_2_vals, Q_12_vals = jnp.array(results)
    param_combinations_python = [tuple(map(float, combo)) for combo in param_combinations]
    multi_index = pd.MultiIndex.from_tuples(param_combinations_python, names=['lr', 'T', 'r_1', 'r_2', 'r_12', 'tau_1', 'tau_2'])
    """df = jl.dump({
        'J_1': list(J_1_vals),
        'J_2': list(J_2_vals),
        'Q_1': list(Q_1_vals),
        'Q_2': list(Q_2_vals),
        'Q_12': list(Q_12_vals)
    }, os.path.join(args.logdir, 'ode.jl'))"""
    df = pd.DataFrame({
        'J_1': list(J_1_vals),
        'J_2': list(J_2_vals),
        'Q_1': list(Q_1_vals),
        'Q_2': list(Q_2_vals),
        'Q_12': list(Q_12_vals)
    }, index=multi_index)

    # Save the DataFrame to a CSV file
    #df.to_csv(os.path.join(args.logdir, 'ode_results.csv'))
    df.to_pickle(os.path.join(args.logdir, 'ode_results.pkl'))

    end = time.time()
    print("Total time taken:", end - start)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ODE Solver")
    parser.add_argument("--logdir",
                        type=str,
                        default='ceph/scratch/npatel/collective_learning/testing')
    parser.add_argument("--D",
                        type=int,
                        default=400,
                        help="Number of dimensions")
    parser.add_argument("--lrs", type=float, nargs='+', default=[1.0], help="Learning rate values")
    parser.add_argument("--Ts", type=int, nargs='+', default=[6], help="Sequence lengths")
    parser.add_argument("--r_1s", type=float, nargs='+', default=[1.0], help="Reward 1 values")
    parser.add_argument("--r_2s", type=float, nargs='+', default=[1.0], help="Reward 2 values")
    parser.add_argument("--r_12s", type=float, nargs='+', default=[1.0], help="Reward 12 values")
    parser.add_argument("--tau_1s", type=float, nargs='+', default=[0.0], help="Tau 1 values")
    parser.add_argument("--tau_2s", type=float, nargs='+', default=[0.0], help="Tau 2 values")
    parser.add_argument("--savepoints", type=int, default=100, help="Number of savepoints")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--R", type=float, default=0.0, help="Initial R value")
    parser.add_argument("--Q_r", type=float, default=1.0, help="Initial Q_r value")
    parser.add_argument("--Q_i", type=float, default=0.0, help="Initial Q_i value")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")

    args = parser.parse_args()

    main_batched(args)
