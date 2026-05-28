#!/usr/bin/env python3
import random
import time
import gymnasium as gym
import numpy as np
import tqdm

# Sarsamax (Q-Learning) algorithm
# Implement training and testing

# Training parameters
n_training_episodes = 100 # 10000
lr = 0.7

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

def initialize_q_table(state_space, action_space):
    """Initialize the Q table"""

    qtable = np.zeros((state_space, action_space))

    return qtable

def greedy_policy(qtable, state):
    "Exploitation: take the action with the highest Q value"""

    action = np.argmax(qtable[state][:])

    return action

def epsilon_greedy_policy(qtable, state, epsilon):
    """Exploitation: take the action with the highest Q value"""

    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)

    if random_num > epsilon:
        # exploitation, pick the action with the highest Q value
        action = greedy_policy(qtable, state)
    else:
        # exploration, pick a random action
        action = env.action_space.sample()

    return action

def train(n_training_episodes,
          min_epsilon,
          max_epsilon,
          decay_rate,
          env,
          max_steps,
          q):
    """Train the Q table"""

    for episode in tqdm.tqdm(range(n_training_episodes)):

        # reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # reset the environment
        state, info = env.reset()

        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(q, state, epsilon)

            # take the action and observe Rt+1 and St+1
            new_state, reward, terminated, truncated, info = env.step(action)

            # sarsamax / q-learning using temporal differences
            q[state][action] = q[state][action] + \
                lr * (reward + gamma * np.max(q[new_state]) - q[state][action])

            if terminated or truncated:
                break

            # we're now in the new state
            state = new_state

    # save q table to file to use in testing
    np.save("qtable.npy", qtable)

    return q

def evaluate():
    """Evaluate the Q table"""

    qtable = np.load("qtable.npy")
    state, info = env.reset()

    for _ in range(max_steps):

        action = greedy_policy(qtable, state)
        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("we're terminated!")
            break

        state = new_state

if __name__ == "__main__":

    # initialize the environment
    env = gym.make("FrozenLake-v1", render_mode="human")

    # reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    print("observation space", env.observation_space)

    state_space = env.observation_space.n
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions\n")

    qtable = initialize_q_table(state_space, action_space)
    print("Q table before training:\n", qtable)

    # qtable = train(
    #     n_training_episodes,
    #     min_epsilon,
    #     max_epsilon,
    #     decay_rate,
    #     env,
    #     max_steps,
    #     qtable)
    # print("Q table after training:\n", qtable)

    evaluate()

    env.close()
