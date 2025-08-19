import numpy as np
import multiprocessing
import argparse
import os
import joblib as jl

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

def Prob_correct(J_1, J_2, Q_1, Q_2, Q_12, S, epsilon):
    P = ((1/2 + 1/jnp.pi * jnp.arcsin(R / (jnp.sqrt((S_r + delta * S_i) * (Q_r + delta * Q_i))))) * 
         (1 - 2 * epsilon) + epsilon)
    return P

#calculates the baseline (i.e the expected reward)
def Baseline(R, Q_r, Q_i, S_r, S_i, r_1, r_2, T, delta, sigma_1, sigma_2, epsilon):
    return (sigma_1 + sigma_2 - 1) * (r_1 + r_2) * Prob_correct(R, Q_r, Q_i, S_r, S_i, delta, epsilon)**T + (1 - sigma_1) * r_1 - sigma_1 * r_2 

"""
functions to calculate expectations of relevant aligning field quantities
"""
def Mod_Nu(S):
    return jnp.sqrt(2 * S/ jnp.pi)

def Mod_1(Q_1):
    return jnp.sqrt(2 * Q_1/ jnp.pi)

def Mod_2(Q_2):
    return jnp.sqrt(2 * Q_2/ jnp.pi)

def Sgn_Nu_1(J_1, S):
    return jnp.sqrt(2 / jnp.pi) * J_1 / jnp.sqrt(S)

def Sgn_Nu_2(J_2, S):
    return jnp.sqrt(2 / jnp.pi) * J_2 / jnp.sqrt(S)

def Sgn_1_2(Q_1, Q_12):
    return jnp.sqrt(2 / jnp.pi) * Q_12 / jnp.sqrt(Q_1)

def Sgn_2_1(Q_2, Q_12):
    return jnp.sqrt(2 / jnp.pi) * Q_12 / jnp.sqrt(Q_2)

def Nu_sgn_1(J_1, Q_1):
    return jnp.sqrt(2 / jnp.pi) * J_1 / jnp.sqrt(Q_1)

def Nu_sgn_2(J_2, Q_2):
    return jnp.sqrt(2 / jnp.pi) * J_2 / jnp.sqrt(Q_2)

def Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (Q_12 * S - J_1 * J_2) / jnp.sqrt((Q_1 * S - J_1**2) * (Q_2 * S - J_2**2))

def Rho_1(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (J_2 * Q_1 - J_1 * Q_12) / jnp.sqrt((Q_1 * S -J_1*2) * (Q_1 * Q_2 - Q_12**2))

def Rho_2(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (J_1 * Q_2 - J_2 * Q_12) / jnp.sqrt((Q_2 * S -J_2*2) * (Q_1 * Q_2 - Q_12**2))

def Heavi_12_Nu(J_1, J_2, Q_1, Q_2, Q_12, S):
    prob = 1/4 + 1/(2 * jnp.pi) * (jnp.arcsin(Q_12 / jnp.sqrt(Q_1*Q_2)) + jnp.arcsin(J_1/jnp.sqrt(Q_1*S)) + jnp.arcsin(J_2/jnp.sqrt(Q_2*S)))
    return prob

def Heavi_12_negheavi_2_Nu(J_1, J_2, Q_1, Q_2, Q_12, S):
    prob = 1/(2 * jnp.pi) * (-jnp.arccos(Q_12 / jnp.sqrt(Q_1*Q_2)) + jnp.arccos(J_1/jnp.sqrt(Q_1*S)) + jnp.arccos(J_2/jnp.sqrt(Q_2*S)))
    return prob

def Heavi_12_sgn_2_Nu(J_1, J_2, Q_1, Q_2, Q_12):
    return (J_1/jnp.sqrt(Q_1) + J_2/jnp.sqrt(Q_2)) * (jnp.sqrt(Q_1*Q_2) - Q_12) / (jnp.pi * jnp.sqrt(Q_1*Q_2 - Q_12**2))

def Heavi_12_mod_Nu(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (J_1 + J_2 + jnp.sqrt(2 * S / jnp.pi) - 
            1/jnp.pi * (jnp.sqrt(2*S/jnp.pi) * jnp.arccos(Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        2 * J_1 * jnp.arccos(Rho_1(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        2 * J_2 * jnp.arccos(Rho_2(J_1, J_2, Q_1, Q_2, Q_12, S))))

def Sgn_1_sgn_2_mod_Nu(J_1, J_2, Q_1, Q_2, Q_12, S):
    return (jnp.sqrt(2 / jnp.pi) * (J_1/jnp.sqrt(Q_1) + J_2/jnp.sqrt(Q_2) + jnp.sqrt(S)) - 
            4/(jnp.pi * jnp.sqrt(2*jnp.pi)) * (J_1/jnp.sqrt(Q_1) * jnp.arccos(Rho_1(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        J_2/jnp.sqrt(Q_2) * jnp.arccos(Rho_2(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        jnp.sqrt(S) * jnp.arccos(Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S))))

def Sgn_2_sgn_Nu_mod_1(J_1, J_2, Q_1, Q_2, Q_12, S):
    return 2 * jnp.sqrt(2) / (jnp.pi * jnp.sqrt(jnp.pi)) * (jnp.sqrt(Q_1) * jnp.arcsin(Rho_1(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        Q_12 / jnp.sqrt(Q_2) * jnp.arcsin(Rho_2(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        J_1 / jnp.sqrt(S) * jnp.arcsin(Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S)))

def Sgn_1_sgn_Nu_mod_2(J_1, J_2, Q_1, Q_2, Q_12, S):
    return 2 * jnp.sqrt(2) / (jnp.pi * jnp.sqrt(jnp.pi)) * (jnp.sqrt(Q_2) * jnp.arcsin(Rho_2(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        Q_12 / jnp.sqrt(Q_1) * jnp.arcsin(Rho_1(J_1, J_2, Q_1, Q_2, Q_12, S)) +
                        J_2 / jnp.sqrt(S) * jnp.arcsin(Rho_12(J_1, J_2, Q_1, Q_2, Q_12, S)))


    
"""
calculate order parameter updates
"""
def dR_dt(T, eta, r_1, r_2, sigma_1, sigma_2, baseline, nu_sgn_lambda, nu_sgn_nu, P):
    dR = eta * ((sigma_1 + sigma_2 - 1) * (r_1 + r_2) / 2 * (nu_sgn_lambda + nu_sgn_nu) * P**(T-1) + 
                ((1 - sigma_1) * r_1 - sigma_1 * r_2 - baseline) * nu_sgn_lambda)
    return dR

def dQr_dt(T, eta, r_1, r_2, rho, sigma_1, sigma_2, baseline, lambdar_sgn_lambda, lambdar_sgn_nu, P):
    dQ_r = (eta * ((sigma_1 + sigma_2 - 1) * (r_1 + r_2) * (lambdar_sgn_lambda + lambdar_sgn_nu) * P**(T-1) + 
                   2 * ((1 - sigma_1) * r_1 - sigma_1 * r_2 - baseline) * lambdar_sgn_lambda) + 
                   eta**2 * (1-jnp.sign(rho) * rho) / T * (((sigma_1 + sigma_2 - 1) * (r_1**2 - r_2**2) - 2 * baseline * (sigma_1 + sigma_2 - 1) * (r_1 + r_2)) * P**T + 
                                  (1 - sigma_1) * r_1**2 + sigma_1 * r_2**2 - 2 * baseline * ((1 - sigma_1) * r_1 - sigma_1 * r_2) + baseline**2))
    return dQ_r

def dQi_dt(T, eta, r_1, r_2, rho, delta, sigma_1, sigma_2, baseline, lambdai_sgn_lambda, P):
    dQ_i = (eta * ((sigma_1 + sigma_2 - 1) * (r_1 + r_2) * lambdai_sgn_lambda * P**(T-1) + 
                   2 * ((1 - sigma_1) * r_1 - sigma_1 * r_2 - baseline) * lambdai_sgn_lambda) + 
                   eta**2 * delta * jnp.heaviside(rho, 0) * rho / T * (((sigma_1 + sigma_2 - 1) * (r_1**2 - r_2**2) - 2 * baseline * (sigma_1 + sigma_2 - 1) * (r_1 + r_2)) * P**T + 
                                               (1 - sigma_1) * r_1**2 + sigma_1 * r_2**2 - 2 * baseline * ((1 - sigma_1) * r_1 - sigma_1 * r_2) + baseline**2))
    return dQ_i

def update_D_times(dt, rho, delta, eta, T, r_1, r_2, R, Q_r, Q_i, S_r, S_i, 
                   sigma_1, sigma_2, epsilon, baseline_bool):
    D = 400
    def body(_, state):
        R, Q_r, Q_i = state

        baseline = baseline_bool * Baseline(R, Q_r, Q_i, S_r, S_i, r_1, r_2, T, delta, sigma_1, sigma_2, epsilon)
        lambdai_sgn_lambda = Lambdai_sgn_lambda(Q_r, Q_i, delta)
        lambdar_sgn_lambda = Lambdar_sgn_lambda(Q_r, Q_i, delta)
        lambdar_sgn_nu = Lambdar_sgn_nu(R, S_r, S_i, delta, epsilon)
        nu_sgn_lambda = Nu_sgn_lambda(R, Q_r, Q_i, delta)
        nu_sgn_nu = Nu_sgn_nu(S_r, S_i, delta, epsilon)
        P = Prob_correct(R, Q_r, Q_i, S_r, S_i, delta, epsilon)

        dR = dR_dt(T, eta, r_1, r_2, sigma_1, sigma_2, baseline, nu_sgn_lambda, nu_sgn_nu, P) * dt
        dQ_r = dQr_dt(T, eta, r_1, r_2, rho, sigma_1, sigma_2, baseline, lambdar_sgn_lambda, lambdar_sgn_nu, P) * dt
        dQ_i = dQi_dt(T, eta, r_1, r_2, rho, delta, sigma_1, sigma_2, baseline, lambdai_sgn_lambda, P) * dt
        dQ_r = 0
        dQ_i = 0

        R = R + dR
        Q_r = Q_r + dQ_r 
        Q_i = Q_i + dQ_i

        return R, Q_r, Q_i

    return lax.fori_loop(0, D, body, (R, Q_r, Q_i))

def lookup_index(x, keys, values):
    match = keys == x + 1
    exists = jnp.any(match)              # bool[]
    first_idx = jnp.argmax(match)        # int[]
    return jnp.where(exists, values[first_idx], -1)

def ODE_solver(dt, rho, delta, eta, T, r_1, r_2, sigma_1, sigma_2, epsilon, baseline_bool, epochs, savepoints):
    
    # Initialize order parameters either from standard values or from simulation values
    """R = 1e-4 + (R - 1e-4) * use_simulation_initialization
    Q_r = (1 - jnp.sign(rho) * rho) + (Q_r - (1 - jnp.sign(rho) * rho)) * use_simulation_initialization
    Q_i = jnp.heaviside(rho, 0) * (rho + (Q_i - rho) * use_simulation_initialization)
    S_r = (1 - jnp.sign(rho) * rho) + (S_r - (1 - jnp.sign(rho) * rho)) * use_simulation_initialization
    S_i = jnp.heaviside(-rho, 0) * (-rho + (S_i + rho) * use_simulation_initialization)"""

    R = 1e-4
    Q_r = (1 - jnp.sign(rho) * rho)
    Q_i = jnp.heaviside(rho, 0) * rho
    S_r = (1 - jnp.sign(rho) * rho)
    S_i = - jnp.heaviside(-rho, 0) * rho

    log_keys = np.unique(np.logspace(0, np.log10(epochs - 1), num=savepoints, dtype=int))
    save_number = len(log_keys)
    log_keys = jnp.array(log_keys)
    log_values = jnp.arange(save_number, dtype=int)

    def body(e, carry):
        R_s, Q_r_s, Q_i_s, R, Q_r, Q_i = carry

        index = lookup_index(e, log_keys, log_values)
        R_s = R_s.at[index].set(R)
        Q_r_s = Q_r_s.at[index].set(Q_r)
        Q_i_s = Q_i_s.at[index].set(Q_i)

        R, Q_r, Q_i = update_D_times(dt, rho, delta, eta, T, r_1, r_2, R, Q_r, Q_i, S_r, S_i, sigma_1, sigma_2, epsilon, baseline_bool)

        return R_s, Q_r_s, Q_i_s, R, Q_r, Q_i

    #create logspaced index array
    #indices = jnp.logspace(0, jnp.log10(time_steps), num=100, dtype=int)
    R_s = jnp.zeros((save_number+1))
    Q_r_s = jnp.zeros((save_number+1))
    Q_i_s = jnp.zeros((save_number+1))

    R_s, Q_r_s, Q_i_s, _, _, _ = lax.fori_loop(
        0, epochs,
        body,
        (R_s, Q_r_s, Q_i_s, R, Q_r, Q_i)
    )

    #return the storing arrays
    return R_s[:-1], Q_r_s[:-1], Q_i_s[:-1]
ODE_solver = jit(ODE_solver, static_argnums=(0, 10, 11, 12))
vmapped_ODE_solver = jax.vmap(ODE_solver, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None))

"""
N-or-more functions
"""
def F_R(T, r_1, r_2, sigma_1, sigma_2, baseline, nu_sgn_lambda, nu_sgn_lambda_heaviside, nu_sgn_lambda_neg_heaviside, alpha, beta):
    r_term = ((sigma_1 + sigma_2 - 1) * (r_1 + r_2) * (alpha * nu_sgn_lambda_heaviside + beta * nu_sgn_lambda_neg_heaviside) / T + 
              ((1 - sigma_1) * r_1 - sigma_1 * r_2 - baseline) * nu_sgn_lambda)
    return r_term

def F_Q(T, r_1, r_2, sigma_1, sigma_2, baseline, lambdar_sgn_lambda, lambdar_sgn_lambda_heaviside, lambdar_sgn_lambda_neg_heaviside, alpha, beta):
    q_term = ((sigma_1 + sigma_2 - 1) * (r_1 + r_2) * (alpha * lambdar_sgn_lambda_heaviside + beta * lambdar_sgn_lambda_neg_heaviside) / T + 
                   ((1 - sigma_1) * r_1 - sigma_1 * r_2 - baseline) * lambdar_sgn_lambda)
    return q_term

def F_Q_2(T, r_1, r_2, rho, sigma_1, sigma_2, baseline, Prob_reward):
    q_2_term = ((1-jnp.sign(rho) * rho) / T * (((sigma_1 + sigma_2 - 1) * (r_1**2 - r_2**2) - 2 * baseline * (sigma_1 + sigma_2 - 1) * (r_1 + r_2)) * Prob_reward + 
                                  (1 - sigma_1) * r_1**2 + sigma_1 * r_2**2 - 2 * baseline * ((1 - sigma_1) * r_1 - sigma_1 * r_2) + baseline**2))
    return q_2_term

def Optimal_learning_rate(R, Q, f_r, f_q, f_q_2):
    opt_lr = 1 / (f_q_2 + 1e-16) * (Q / R * f_r - f_q)
    return opt_lr

def dR_n_more(eta, f_r):
    dR = eta * f_r
    return dR

def dQr_n_more(eta, f_q, f_q_2):
    dQ_r = eta * 2 * f_q + eta**2 * f_q_2
    return dQ_r

def dQi_n_more(eta, T, r_1, r_2, rho, delta, sigma_1, sigma_2, baseline, lambdai_sgn_lambda, lambdai_sgn_lambda_heaviside, lambdai_sgn_lambda_neg_heaviside, alpha, beta, Prob_reward):
    dQ_i = (eta * (2 * (sigma_1 + sigma_2 - 1) * (r_1 + r_2) * (alpha * lambdai_sgn_lambda_heaviside + beta * lambdai_sgn_lambda_neg_heaviside) / T + 
                   2 * ((1 - sigma_1) * r_1 - sigma_1 * r_2 - baseline) * lambdai_sgn_lambda) + 
                   eta**2 * delta * jnp.heaviside(rho, 0) * rho / T * (((sigma_1 + sigma_2 - 1) * (r_1**2 - r_2**2) - 2 * baseline * (sigma_1 + sigma_2 - 1) * (r_1 + r_2)) * Prob_reward + 
                                               (1 - sigma_1) * r_1**2 + sigma_1 * r_2**2 - 2 * baseline * ((1 - sigma_1) * r_1 - sigma_1 * r_2) + baseline**2))
    return dQ_i

def N_mask(n, T):
    return jnp.where(jnp.arange(T+1) < n, 0, 1)

"""
helper functions, for coefficients etc for n or more
"""
def P_powers(P, T):
    return jnp.array([P**i for i in range(T+1)])

def P_powers_reversed(P, T):
    return jnp.array([P**(T - i) for i in range(T+1)])

def Alpha(P, v, i, P_vec, reverse_P_vec, n_mask):
    return jnp.sum(v * i * P_vec * reverse_P_vec * n_mask) / P

def Beta(alpha, P, T, prob_reward):
    return 1 / (1-P) * (T * prob_reward - P * alpha)

def Prob_reward(v, P_vec, reverse_P_vec, n_mask):
    return jnp.sum(v * P_vec * reverse_P_vec * n_mask)

def Binom_coeff(N_static, r_python_int):
    r_eff = min(r_python_int, N_static - r_python_int)
    terms_start = N_static - r_eff + 1
    terms_stop = N_static + 1
    # Ensure terms are int32 for integer product
    terms = jnp.arange(terms_start, terms_stop, dtype=jnp.int32)
    # denominator: r_eff * (r_eff-1) * ... * 1
    den_start = 1
    den_stop = r_eff + 1
    # Ensure denominator terms are int32
    denominator = jnp.arange(den_start, den_stop, dtype=jnp.int32)
    # jnp.prod of an empty array is 1 (as float by default, but int if dtype specified in arange)
    # So if r_eff is 0, terms and denominator are empty, prods are 1. Result 1 // 1 = 1. Correct.
    numerator_prod = jnp.prod(terms)
    denominator_prod = jnp.prod(denominator)
    # Ensure integer division
    return numerator_prod // denominator_prod

def All_coeffs(N_static_val):
    coeffs_array = jnp.zeros((N_static_val + 1,), dtype=jnp.int32)

    # This Python loop will be unrolled by JAX because N_static_val is static.
    # 'r' will be a concrete Python integer in each unrolled iteration.
    for r_val in range(N_static_val + 1):
        coeff = Binom_coeff(N_static_val, r_val)
        coeffs_array = coeffs_array.at[r_val].set(coeff)
    return coeffs_array

"""
update D times for n or more
"""
def update_D_times_n_more(dt, rho, delta, eta, T, r_1, r_2, R, Q_r, Q_i, S_r, S_i, sigma_1, sigma_2, epsilon, baseline_bool, v, i, n_mask, optimal_LR_bool):
    D = 400
    def body(_, state):
        R, Q_r, Q_i = state

        baseline = baseline_bool * Baseline(R, Q_r, Q_i, S_r, S_i, r_1, r_2, T, delta, sigma_1, sigma_2, epsilon)
        lambdai_sgn_lambda = Lambdai_sgn_lambda(Q_r, Q_i, delta)
        lambdar_sgn_lambda = Lambdar_sgn_lambda(Q_r, Q_i, delta)
        nu_sgn_lambda = Nu_sgn_lambda(R, Q_r, Q_i, delta)
        nu_sgn_lambda_heaviside = Nu_sgn_lambda_heaviside(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
        lambdar_sgn_lambda_heaviside = Lambdar_sgn_lambda_heaviside(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
        lambdai_sgn_lambda_heaviside = Lambdai_sgn_lambda_heaviside(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
        lambdar_sgn_lambda_neg_heaviside = Lambdar_sgn_lambda_neg_heaviside(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
        lambdai_sgn_lambda_neg_heaviside = Lambdai_sgn_lambda_neg_heaviside(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
        nu_sgn_lambda_neg_heaviside = Nu_sgn_lambda_neg_heaviside(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
        P = Prob_correct(R, Q_r, Q_i, S_r, S_i, delta, epsilon)

        p_powers = P_powers(P, T)
        reverse_p_powers = P_powers_reversed(1-P, T)

        prob_reward = Prob_reward(v, p_powers, reverse_p_powers, n_mask)
        alpha = Alpha(P, v, i, p_powers, reverse_p_powers, n_mask)
        beta = Beta(alpha, P, T, prob_reward)

        f_r = F_R(T, r_1, r_2, sigma_1, sigma_2, baseline, nu_sgn_lambda, nu_sgn_lambda_heaviside, nu_sgn_lambda_neg_heaviside, alpha, beta)
        f_q = F_Q(T, r_1, r_2, sigma_1, sigma_2, baseline, lambdar_sgn_lambda, lambdar_sgn_lambda_heaviside, lambdar_sgn_lambda_neg_heaviside, alpha, beta)
        f_q_2 = F_Q_2(T, r_1, r_2, rho, sigma_1, sigma_2, baseline, P)

        opt_lr = Optimal_learning_rate(R, Q_r, f_r, f_q, f_q_2)
        lr = eta + (opt_lr - eta) * optimal_LR_bool

        dR = dR_n_more(lr, f_r) * dt
        dQ_r = dQr_n_more(lr, f_q, f_q_2) * dt
        dQ_i = dQi_n_more(lr, T, r_1, r_2, rho, delta, sigma_1, sigma_2, baseline, lambdai_sgn_lambda, lambdai_sgn_lambda_heaviside, lambdai_sgn_lambda_neg_heaviside, alpha, beta, prob_reward) * dt

        R = R + dR
        Q_r = Q_r + dQ_r 
        Q_i = Q_i + dQ_i

        return R, Q_r, Q_i

    return lax.fori_loop(0, D, body, (R, Q_r, Q_i))

def ODE_solver_N_more(dt, rho, delta, eta, T, n, r_1, r_2, sigma_1, sigma_2, epsilon,
                baseline_bool, epochs, savepoints, optimal_LR_bool):
    v = All_coeffs(T)
    i = jnp.arange(T + 1)
    n_mask = N_mask(n, T)
    #i = jnp.array(i)
    #n_mask = jnp.array(n_mask)
    R = 1e-4
    Q_r = (1 - jnp.sign(rho) * rho)
    Q_i = jnp.heaviside(rho, 0) * rho
    S_r = (1 - jnp.sign(rho) * rho)
    S_i = - jnp.heaviside(-rho, 0) * rho

    # Initialize order parameters either from standard values or from simulation values
    """R = 1e-4 + (R - 1e-4) * use_simulation_initialization
    Q_r = (1 - jnp.sign(rho) * rho) + (Q_r - (1 - jnp.sign(rho) * rho)) * use_simulation_initialization
    Q_i = jnp.heaviside(rho, 0) * (rho + (Q_i - rho) * use_simulation_initialization)
    S_r = (1 - jnp.sign(rho) * rho) + (S_r - (1 - jnp.sign(rho) * rho)) * use_simulation_initialization
    S_i = jnp.heaviside(-rho, 0) * (-rho + (S_i + rho) * use_simulation_initialization)"""
    
    log_keys = np.unique(np.logspace(0, np.log10(epochs - 1), num=savepoints, dtype=int))
    save_number = len(log_keys)
    log_keys = jnp.array(log_keys)
    log_values = jnp.arange(save_number, dtype=int)

    def body(e, carry):
        R_s, Q_r_s, Q_i_s, R, Q_r, Q_i = carry

        index = lookup_index(e, log_keys, log_values)
        R_s = R_s.at[index].set(R)
        Q_r_s = Q_r_s.at[index].set(Q_r)
        Q_i_s = Q_i_s.at[index].set(Q_i)

        R, Q_r, Q_i = update_D_times_n_more(dt, rho, delta, eta, T, r_1, r_2, R, Q_r, Q_i, S_r, S_i, 
                                     sigma_1, sigma_2, epsilon, baseline_bool, v, i, n_mask, optimal_LR_bool)

        return R_s, Q_r_s, Q_i_s, R, Q_r, Q_i

    #create logspaced index array
    #indices = jnp.logspace(0, jnp.log10(time_steps), num=100, dtype=int)
    R_s = jnp.zeros((save_number+1))
    Q_r_s = jnp.zeros((save_number+1))
    Q_i_s = jnp.zeros((save_number+1))

    R_s, Q_r_s, Q_i_s, _, _, _ = lax.fori_loop(
        0, epochs,
        body,
        (R_s, Q_r_s, Q_i_s, R, Q_r, Q_i)
    )

    #return the storing arrays
    return R_s[:-1], Q_r_s[:-1], Q_i_s[:-1]

ODE_solver_N_more = jit(ODE_solver_N_more, static_argnums=(0, 4, 11, 12, 13, 14))
vmapped_ODE_solver_N_more = jax.vmap(ODE_solver_N_more, in_axes=(None, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, None, None, None, None))

def main(args):
    os.makedirs(args.logdir, exist_ok=True)

    dt = 1 / args.D
    start = time.time()

    for T in args.Ts:
        for n in args.ns:
            for lr in args.lrs:
                for rho in args.rhos:
                    for r_1 in args.r_1s:
                        for delta in args.deltas:
                            for r_2 in args.r_2s:
                                for epsilon in args.epsilons:
                                    for sigma_1 in args.sigma_1s:
                                        for sigma_2 in args.sigma_2s:
                                            R_s, Q_r_s, Q_i_s = ODE_solver(dt, rho, delta, lr, T, r_1, r_2, sigma_1,
                                                                            sigma_2, epsilon, args.baseline, args.epochs, args.savepoints)
                                            #convert from jax.numpy arrays to numpy array then store the arrays in the dictionary

                                            folder = f'T{T}r_2{r_2}rho{rho}delta{delta}sigma1{sigma_1}sigma2{sigma_2}epsilon{epsilon}baseline{args.baseline}'  
                                            log_folder = os.path.join(args.logdir, folder)
                                            if not os.path.isdir(log_folder):
                                                os.makedirs(log_folder)

                                            with open(os.path.join(log_folder, 'args.json'), 'w') as f:
                                                args_dict = {}
                                                args_dict['rho']=rho
                                                args_dict['delta']=delta
                                                args_dict['lr']=lr
                                                args_dict['T']=T
                                                args_dict['r_1']=r_1
                                                args_dict['r_2']=r_2
                                                args_dict['sigma_1']=sigma_1
                                                args_dict['sigma_2']=sigma_2
                                                args_dict['epsilon']=epsilon
                                                args_dict['baseline']=args.baseline
                                                args_dict['epochs']=args.epochs
                                                args_dict['savepoints']=args.savepoints
                                                json.dump(vars(args), f)
                                            R_s = np.array(R_s)
                                            Q_r_s = np.array(Q_r_s)
                                            Q_i_s = np.array(Q_i_s)
                                            
                                            np.savez(os.path.join(log_folder, 'ode_results.npz'),
                                                    R_s=R_s,
                                                    Q_r_s=Q_r_s,
                                                    Q_i_s=Q_i_s)
    end = time.time()
    print("Total time taken:", end - start)


def main_batched(args):
    dt = 1 / args.D
    os.makedirs(args.logdir, exist_ok=True)

    #create batch of parameter combinations
    param_lists = [args.rhos, args.deltas, args.lrs, args.Ts, args.r_1s, args.r_2s, 
                   args.sigma_1s, args.sigma_2s, args.epsilons]
    param_combinations = list(itertools.product(*param_lists))
    param_array = jnp.array(param_combinations)
    rho_batch, delta_batch, lr_batch, T_batch, r_1_batch, r_2_batch, sigma_1_batch, sigma_2_batch, epsilon_batch = param_array.T

    #run batch of ODEs
    start = time.time()
    results = vmapped_ODE_solver(dt, rho_batch, delta_batch, lr_batch, T_batch, r_1_batch, r_2_batch, sigma_1_batch,
                                 sigma_2_batch, epsilon_batch, args.baseline, args.epochs, args.savepoints)
    
    #store results in pandas dataframe
    R_vals, Qr_vals, Qi_vals, overlaps = jnp.array(results)
    param_combinations_python = [tuple(map(float, combo)) for combo in param_combinations]
    multi_index = pd.MultiIndex.from_tuples(param_combinations_python, names=['rho', 'delta', 'lr', 'T', 'r_1', 'r_2', 'sigma_1', 'sigma_2', 'epsilon'])
    """df = jl.dump({
        'R': list(R_vals),
        'Qr': list(Qr_vals),
        'Qi': list(Qi_vals)
    }, os.path.join(args.logdir, 'ode.jl'))"""
    df = pd.DataFrame({
        'R': list(R_vals),
        'Qr': list(Qr_vals),
        'Qi': list(Qi_vals),
        'overlap': list(overlaps)
    }, index=multi_index)

    # Save the DataFrame to a CSV file
    #df.to_csv(os.path.join(args.logdir, 'ode_results.csv'))
    df.to_pickle(os.path.join(args.logdir, 'ode_results.pkl'))

    end = time.time()
    print("Total time taken:", end - start)
    
def main_batched_N_more(args):
    dt = 1 / args.D
    os.makedirs(args.logdir, exist_ok=True)

    #create batch of parameter combinations
    param_lists = [args.rhos, args.deltas, args.lrs, args.ns, args.r_1s, args.r_2s, 
                   args.sigma_1s, args.sigma_2s, args.epsilons]
    
    param_combinations = list(itertools.product(*param_lists))
    param_array = jnp.array(param_combinations)
    rho_batch, delta_batch, lr_batch, n_batch, r_1_batch, r_2_batch, sigma_1_batch, sigma_2_batch, epsilon_batch = param_array.T

    #run batch of ODEs
    start = time.time()
    results = vmapped_ODE_solver_N_more(dt, rho_batch, delta_batch, lr_batch, args.Ts[0], n_batch, r_1_batch, r_2_batch, sigma_1_batch,
                                 sigma_2_batch, epsilon_batch, args.baseline, args.epochs, args.savepoints, args.optimal_LR)
    
    #store results in pandas dataframe
    R_vals, Qr_vals, Qi_vals = jnp.array(results)
    param_combinations_python = [tuple(map(float, combo)) for combo in param_combinations]
    multi_index = pd.MultiIndex.from_tuples(param_combinations_python, names=['rho', 'delta', 'lr', 'n', 'r_1', 'r_2', 'sigma_1', 'sigma_2', 'epsilon'])
    """df = jl.dump({
        'R': list(R_vals),
        'Qr': list(Qr_vals),
        'Qi': list(Qi_vals)
    }, os.path.join(args.logdir, 'ode.jl'))"""
    df = pd.DataFrame({
        'R': list(R_vals),
        'Qr': list(Qr_vals),
        'Qi': list(Qi_vals)
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
                        default='ceph/scratch/npatel/le_cake/testing')
    parser.add_argument("--D",
                        type=int,
                        default=400,
                        help="Number of dimensions")
    parser.add_argument("--rhos", type=float, nargs='+', default=[0.0], help="Rho values")
    parser.add_argument("--deltas", type=float, nargs='+', default=[0.0], help="Delta values")
    parser.add_argument("--lrs", type=float, nargs='+', default=[1.0], help="Learning rate values")
    parser.add_argument("--Ts", type=int, nargs='+', default=[6], help="Sequence lengths")
    parser.add_argument("--ns", type=int, nargs='+', default=[6], help="Sequence lengths")
    parser.add_argument("--r_1s", type=float, nargs='+', default=[1.0], help="Reward 1 values")
    parser.add_argument("--r_2s", type=float, nargs='+', default=[0.0], help="Reward 2 values")
    parser.add_argument("--sigma_1", type=float, nargs='+', default=[1.0], help="Sigma 1 values")
    parser.add_argument("--sigma_2", type=float, nargs='+', default=[1.0], help="Sigma 2 values")
    parser.add_argument("--epsilon", type=float, nargs='+', default=[0.0], help="Epsilon 2 values")
    parser.add_argument("--R", type=float, default=0.0, help="Initial R value")
    parser.add_argument("--Q_r", type=float, default=1.0, help="Initial Q_r value")
    parser.add_argument("--Q_i", type=float, default=0.0, help="Initial Q_i value")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--baseline", action="store_true", help="Use baseline")

    args = parser.parse_args()

    main(args)
