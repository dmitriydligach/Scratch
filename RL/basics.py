#!/usr/bin/env python3
import random
import time
import gymnasium as gym
import numpy as np
import tqdm

# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7  # Learning rate

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

    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    "Exploitation: take the action with the highest Q value"""

    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1

    random_num = random.uniform(0, 1)

    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    """Train the Q table"""

    for episode in tqdm.tqdm(range(n_training_episodes)):

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Reset the environment
        state, info = env.reset()

        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state
    return Qtable

# Initialize the environment
env = gym.make("FrozenLake-v1", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

print("observation space", env.observation_space)
print()

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")
print()

Qtable_frozenlake = initialize_q_table(state_space, action_space)

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

# for _ in range(1000):
#     # this is where you would insert your policy
#     action = env.action_space.sample()
#
#     # pause
#     time.sleep(2)
#
#     # step (transition) through the environment with the action
#     # receiving the next observation, reward and if the episode has terminated or truncated
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     print("action: ", action)
#     print("observation: ", observation)
#     print("reward: ", reward)
#     print()
#
#     # If the episode has ended then we can reset to start a new episode
#     if terminated or truncated:
#         observation, info = env.reset()

env.close()
