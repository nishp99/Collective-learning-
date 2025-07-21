import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
import jax
import argparse
"""
student is dim (D)
teacher is dim (D)
x_s is dim (T,D)
y_s is dim (T)
G_s is dim (T)

lr is dim (1)
D is dim (1)
T is dim (1)
gamma is dim (1)

a is dim (1)
b is dim (1)
rho is dim (1)
"""

"""
create initial n students and teacher
"""

def create_student(dim, n, key):
    student = random.normal(key, shape=(dim))
    students = jnp.tile(student, (n, 1))
    return students

def create_teacher(relevant_dim, key):
    teacher = random.normal(key, shape=(relevant_dim))
    teacher = teacher / jnp.linalg.norm(teacher) * jnp.sqrt(relevant_dim)
    return teacher

"""
calculate order parameters
"""
def order_parameters(students, teacher, D, index):
    R = jnp.einsum('n D, D -> n', students[:, :index], teacher[:index]) / D
    Q_r = jnp.einsum('n D, n D -> n', students[:, :index], students[:, :index]) / D
    Q_i = jnp.einsum('n D, n D -> n', students[:, index:], students[:, index:]) / D
    return R, Q_r, Q_i

#order_parameters = jit(order_parameters, static_argnums=(3))

"""
create population covariance
"""
def population_covariance(D, delta, index):
    covariance_matrix = np.zeros((D, D))
    #index = int(jnp.round((1-rho) * D))
    covariance_matrix[:index, :index] = jnp.eye(index)
    covariance_matrix[index:, index:] = jnp.eye(D - index) * delta
    return covariance_matrix

"""
randomly initialize seed
"""
def random_seed():
    seed = np.random.randint(0, 2**32 - 1)
    return seed

"""
create random key
"""
def create_key(seed):
    key = random.PRNGKey(seed)
    return key

"""
sample obervations for a discrete amount of time for all parallel students for entire episodes
"""
def sample_observations(T, D, n, Covariance, key):
    X_s = random.multivariate_normal(key, mean=jnp.zeros(Covariance.shape[0]), cov=Covariance, shape=(n, T, D))
    return X_s
"""
sample reward noise for a discrete amount of time for all parallel students for entire episodes
"""
def sample_reward_noise(D, n, sigma, key):
    Epsilon_s = random.bernoulli(key, p=sigma, shape=(n, D)).astype(jnp.int32)
    return Epsilon_s

def sample_teacher_noise(T, D, n, epsilon, key):
    Z_s = 2 * random.bernoulli(key, p=1-epsilon, shape=(n, T, D)).astype(jnp.int32) - 1
    return Z_s

"""
sample create teacher decisions
"""
def teacher_decisions(teacher, X_s, Z_s, index):
    Y_t_s = jnp.sign(jnp.einsum('D, n T J D -> n T J', teacher, X_s[:, :, :, :index])) * Z_s
    return Y_t_s

"""
calculate baseline (assume perfect value function)
"""
def Prob_correct_sim(R, Q_r, Q_i, S_r, S_i, delta, epsilon):
    P = (1/2 + 1/jnp.pi * jnp.arcsin(R / jnp.sqrt((S_r + delta * S_i) * (Q_r + delta * Q_i)))) * (1 - 2 * epsilon) + epsilon
    return P


def calculate_baseline(students, teacher, r_1, r_2, delta, sigma_1, sigma_2, epsilon, S_r, S_i, D, T, index):
    R, Q_r, Q_i = order_parameters(students, teacher, D, index)
    Prob = Prob_correct_sim(R, Q_r, Q_i, S_r, S_i, delta, epsilon)
    baseline = (sigma_1 + sigma_2 - 1) * (r_1 + r_2) * Prob**T + (1 - sigma_1) * r_1 - sigma_1 * r_2
    return baseline

"""
actions
"""
def student_actions(students, x_s):
    y_s = jnp.sign(jnp.einsum('n D, n T D -> n T', students, x_s))
    return y_s

def teacher_actions(teacher, x_s, z_s, index):
    y_t = jnp.sign(jnp.einsum('D, n T D -> n T', teacher, x_s[:, :, :index])) * z_s
    return y_t

"""
example reward function (lets choose extended Hebbian for now)
lets lose the discounted reward for now, as discount factor is 1. and hard code the rewards
"""
def rl_perceptron_ALL(y_s, y_t, r_1, r_2, epsilon_1_s, epsilon_2_s, baseline):
    reward = (jnp.all(y_s == y_t, axis=1).astype(jnp.float32) * (epsilon_1_s + epsilon_2_s - 1) * (r_1 + r_2) + 
              (1 - epsilon_1_s) * r_1 - epsilon_1_s * r_2 - baseline)
    #rewards = jnp.zeros_like(y_s, dtype=jnp.float32)
    #rewards = rewards.at[:, -1].set(reward)
    rewards = jnp.ones_like(y_s, dtype=jnp.float32)
    rewards = rewards * reward[:, None]
    return rewards

def rl_perceptron_NMORE(y_s, y_t, r_1, r_2, T, n, epsilon_1_s, epsilon_2_s, baseline):
    reward = (jnp.where(jnp.sum(y_s * y_t, axis=1) >= 2*n - T , 1, 0).astype(jnp.float32) * (epsilon_1_s + epsilon_2_s - 1) * (r_1 + r_2) + 
              (1 - epsilon_1_s) * r_1 - epsilon_1_s * r_2 - baseline)
    #rewards = jnp.zeros_like(y_s, dtype=jnp.float32)
    #rewards = rewards.at[:, -1].set(reward)
    rewards = jnp.ones_like(y_s, dtype=jnp.float32)
    rewards = rewards * reward[:, None]
    return rewards
    
"""
discounted reward (mediate for future reward)
"""
def discounted_reward(rewards, discount, T):
    discounted_future_rewards = jnp.zeros_like(rewards, dtype=jnp.float32)
    cumulative_reward = 0
    for t in range(T - 1, -1, -1):
        cumulative_reward = rewards[:, t] + discount * cumulative_reward
        discounted_future_rewards = discounted_future_rewards.at[:, t].set(cumulative_reward)

    return discounted_future_rewards

"""
update equation
"""
@jit
def student_update(y_s, x_s, G_s, lr, D, T):
    update = lr / (T * jnp.sqrt(D)) * jnp.einsum('n T, n T D -> n D',(y_s * G_s), x_s)
    return update

"""def update_student_D_times(students, teacher, X_s, Epsilon_s, Z_s, D, T, index, lr, r_1, r_2):
    for i in range(D):
        x_s = X_s[:, :, i, :]
        epsilon_s = Epsilon_s[:, i]
        z_s = Z_s[:, :, i]

        y_s = student_actions(students, x_s)
        y_t = teacher_actions(teacher, x_s, z_s, index)
        G_s = rl_perceptron_Hebbian(y_s, y_t, r_1, r_2, epsilon_s)

        students = students + student_update(y_s, x_s, G_s, lr, D, T)

    return students"""
    
#update_student_D_times = jit(update_student_D_times, static_argnums=(3, 4, 5))

def update_student_D_times(students, X_s, Epsilon_1_s, Epsilon_2_s, Y_t_s, D, T, n, lr, r_1, r_2, baseline_bool, 
                           teacher, delta, sigma_1, sigma_2, epsilon, S_r, S_i, student_index, min_index):

    def body_fn(i, students):
        x_s = X_s[:, :, i, :student_index]

        epsilon_1_s = Epsilon_1_s[:, i]
        epsilon_2_s = Epsilon_2_s[:, i]
        y_t = Y_t_s[:, :, i]

        baseline = baseline_bool * calculate_baseline(students, teacher, r_1, r_2, delta, sigma_1, sigma_2, 
                                                      epsilon, S_r, S_i, D, T, min_index)
        
        y_s = student_actions(students, x_s)
        #y_t = teacher_actions(teacher, x_s, z_s, index)
        #G_s = rl_perceptron_ALL(y_s, y_t, r_1, r_2, epsilon_1_s, epsilon_2_s, baseline)
        G_s = rl_perceptron_NMORE(y_s, y_t, r_1, r_2, T, n, epsilon_1_s, epsilon_2_s, baseline)

        students = students + student_update(y_s, x_s, G_s, lr, D, T)

        return students

    students = lax.fori_loop(0, D, body_fn, students)
    return students


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    sample_observations = jit(sample_observations, static_argnums=(0, 1, 2))
    sample_reward_noise = jit(sample_reward_noise, static_argnums=(0, 1))
    sample_teacher_noise = jit(sample_teacher_noise, static_argnums=(0, 1, 2))
    teacher_decisions = jit(teacher_decisions, static_argnums=(3))
    teacher_actions = jit(teacher_actions, static_argnums=(3))
    discounted_reward = jit(discounted_reward, static_argnums=(2))
    update_student_D_times = jit(update_student_D_times, static_argnums=(5, 6, 7, 19, 20))
