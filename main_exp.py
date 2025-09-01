import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
import jax
import argparse

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
create initial n students and teacher
"""
def create_student(dim, n, key):
    student = random.normal(key, shape=(dim))
    students = jnp.tile(student, (n, 1))
    return students

def create_teacher(dim, key):
    teacher = random.normal(key, shape=(dim))
    teacher = teacher / jnp.linalg.norm(teacher) * jnp.sqrt(dim)
    return teacher

"""
calculate order parameters
"""

def order_params_ode(students_1, students_2, teacher, D):
    J_1 = jnp.einsum('n D, D -> n', students_1, teacher) / D
    J_2 = jnp.einsum('n D, D -> n', students_2, teacher) / D
    Q_1 = jnp.einsum('n D, n D -> n', students_1, students_1) / D
    Q_2 = jnp.einsum('n D, n D -> n', students_2, students_2) / D
    Q_12 = jnp.einsum('n D, n D -> n', students_1, students_2) / D
    return J_1, J_2, Q_1, Q_2, Q_12

"""
sample obervations for a discrete amount of time for all parallel students for entire episodes
"""
#@jit
def sample_observations(T, D, n, key):
    X_s = random.multivariate_normal(key, mean=jnp.zeros(D), cov=jnp.eye(D), shape=(n, T, D))
    return X_s

"""
sample create teacher decisions
"""
#@jit
def teacher_decisions(teacher, X_s):
    Y_t_s = jnp.sign(jnp.einsum('D, n T J D -> n T J', teacher, X_s))
    return Y_t_s

"""
actions
"""
def student_actions(students, x_s):
    y_s = jnp.sign(jnp.einsum('n D, n T D -> n T', students, x_s))
    return y_s
"""
example reward function (lets choose extended Hebbian for now)
lets lose the discounted reward for now, as discount factor is 1. and hard code the rewards
"""
def Individual_ALL(y_s, y_t):
    reward = jnp.all(y_s == y_t, axis=1).astype(jnp.float32)
    #rewards = jnp.zeros_like(y_s, dtype=jnp.float32)
    #rewards = rewards.at[:, -1].set(reward)
    #rewards = jnp.ones_like(y_s, dtype=jnp.float32)
    #rewards = rewards * reward[:, None]
    return reward

def Collaborative_ALL(y_1, y_2, y_t):
    reward = (jnp.all(y_1 == y_2, axis=1) * jnp.all(y_1 == y_t, axis=1)).astype(jnp.float32)
    return reward

def Reward(r_1, r_2, r_12, tau_1, tau_2, y_1, y_2, y_t):
    collaborative_reward = r_12 * Collaborative_ALL(y_1, y_2, y_t)
    reward_1 = r_1 * Individual_ALL(y_1, y_t)
    reward_2 = r_2 * Individual_ALL(y_2, y_t)

    R_1 = reward_1 + tau_2 * reward_2 + collaborative_reward
    R_2 = reward_2 + tau_1 * reward_1 + collaborative_reward
    return R_1, R_2
    
"""
update equation
"""
def student_update(y_s, x_s, R, lr, D, T):
    update = lr / (T * jnp.sqrt(D)) * jnp.einsum('n T, n T D -> n D',(y_s * R[:, jnp.newaxis]), x_s)
    return update

def Update_student_D_times(teacher, students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12, 
                           X_s, Y_t_s, D, T, lr, r_1, r_2, r_12, tau_1, tau_2):

    def body(i, carry):
        students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12 = carry
        x_s = X_s[:, :, i, :]
        y_t = Y_t_s[:, :, i]
        
        y_1_s = student_actions(students_1, x_s)
        y_2_s = student_actions(students_2, x_s)
        R_1_s, R_2_s = Reward(r_1, r_2, r_12, tau_1, tau_2, y_1_s, y_2_s, y_t)

        students_1 = students_1 + student_update(y_1_s, x_s, R_1_s, lr, D, T)
        students_2 = students_2 + student_update(y_2_s, x_s, R_2_s, lr, D, T)

        J_1, J_2, Q_1, Q_2, Q_12 = order_params_ode(students_1, students_2, teacher, D)

        return students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12

    return lax.fori_loop(0, D, body, (students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12))
#Update_student_D_times = jit(Update_student_D_times, static_argnums=(4,5,6,7,8,9,10,11))

"""
jittable way for finding corresponding index
"""
def lookup_index(x, keys, values):
    match = keys == x + 1
    exists = jnp.any(match)              # bool[]
    first_idx = jnp.argmax(match)        # int[]
    return jnp.where(exists, values[first_idx], -1)

"""
run the simulation for single set of parameters and given number of epochs, 
and store the order parameters at logarithmically spaced intervals
"""
def Run_simulation(D, n, lr, teacher, students_1, students_2, T, r_1, r_2, r_12, tau_1, tau_2, epochs, savepoints):

    log_keys = np.unique(np.logspace(0, np.log10(epochs - 1), num=savepoints, dtype=int))
    save_number = len(log_keys)
    log_keys = jnp.array(log_keys)
    log_values = jnp.arange(save_number, dtype=int)

    save_epochs = np.unique(np.logspace(0, np.log10(epochs - 1), num=savepoints, dtype=int))
    save_epochs = jnp.array(save_epochs)
    deltas = jnp.diff(save_epochs, prepend=0)  # number of steps between savepoints
    num_saves = save_epochs.shape[0]

    J_1_s = jnp.zeros((num_saves+1, n))
    J_2_s = jnp.zeros((num_saves+1, n))
    Q_1_s = jnp.zeros((num_saves+1, n))
    Q_2_s = jnp.zeros((num_saves+1, n))
    Q_12_s = jnp.zeros((num_saves+1, n))

    J_1, J_2, Q_1, Q_2, Q_12 = order_params_ode(students_1, students_2, teacher, D)

    J_1_s = J_1_s.at[0, :].set(J_1)
    J_2_s = J_2_s.at[0, :].set(J_2)
    Q_1_s = Q_1_s.at[0, :].set(Q_1)
    Q_2_s = Q_2_s.at[0, :].set(Q_2)
    Q_12_s = Q_12_s.at[0, :].set(Q_12)

    for i in range(len(deltas)):
        for _ in range(deltas[i]):
            x_key = create_key(random_seed())
            X_s = sample_observations(T, D, n, x_key)
            Y_t_s = teacher_decisions(teacher, X_s)

            students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12 = Update_student_D_times(teacher, students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12, 
                                                                                     X_s, Y_t_s, D, T, lr, r_1, r_2, r_12, tau_1, tau_2)
            
            J_1_s = J_1_s.at[i, :].set(J_1)
            J_2_s = J_2_s.at[i, :].set(J_2)
            Q_1_s = Q_1_s.at[i, :].set(Q_1)
            Q_2_s = Q_2_s.at[i, :].set(Q_2)
            Q_12_s = Q_12_s.at[i, :].set(Q_12)


    """for epoch in range(epochs):
        index = lookup_index(epoch, log_keys, log_values)
        J_1_s = J_1_s.at[index, :].set(J_1)
        J_2_s = J_2_s.at[index, :].set(J_2)
        Q_1_s = Q_1_s.at[index, :].set(Q_1)
        Q_2_s = Q_2_s.at[index, :].set(Q_2)
        Q_12_s = Q_12_s.at[index, :].set(Q_12)

        x_key = create_key(random_seed())
        X_s = sample_observations(T, D, n, x_key)
        Y_t_s = teacher_decisions(teacher, X_s)

        students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12 = Update_student_D_times(teacher, students_1, students_2, J_1, J_2, Q_1, Q_2, Q_12, 
                                                                                 X_s, Y_t_s, D, T, lr, r_1, r_2, r_12, tau_1, tau_2)"""
        
        
    #return J_1_s[:save_number, :], J_2_s[:save_number, :], Q_1_s[:save_number, :], Q_2_s[:save_number, :], Q_12_s[:save_number, :]
    return J_1_s, J_2_s, Q_1_s, Q_2_s, Q_12_s

"""def main(args):
    teacher = create_teacher(D, create_key(random_seed()))
    student_1 = create_student(D, n, create_key(random_seed()))
    student_2 = create_student(D, n, create_key(random_seed()))"""



"""if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    boogit boogity boogity
"""
    
