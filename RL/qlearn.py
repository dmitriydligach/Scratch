#!/usr/bin/env python3
import random
import time
import gymnasium as gym
import numpy as np
import tqdm

# Sarsamax (Q-Learning) algorithm
# Implement training and testing

# Training parameters
n_training_episodes = 10000
lr = 0.7

# Evaluation parameters
n_eval_episodes = 25  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

def initialize_q_table(state_space, action_space):
    """Initialize the Q table to zeros"""

    return np.zeros((state_space, action_space))

def greedy_policy(qtable, state):
    """Return action with the highest Q value"""

    return np.argmax(qtable[state][:])

def epsilon_greedy_policy(env, qtable, state, epsilon):
    """Exploration with probability epsilon"""

    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)

    if random_num > epsilon:
        # exploitation, pick the action with the highest Q value
        action = greedy_policy(qtable, state)
    else:
        # exploration, pick a random action
        action = env.action_space.sample()

    return action

def train(env,
          n_training_episodes,
          min_epsilon,
          max_epsilon,
          decay_rate,
          max_steps,
          q):
    """Train the Q table"""

    for episode in tqdm.tqdm(range(n_training_episodes)):

        # reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # no seed to get a different trajectory every time
        state, info = env.reset()

        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):

            # choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(env, q, state, epsilon)

            # take the action and observe Rt+1 and St+1
            new_state, reward, terminated, truncated, info = env.step(action)

            # print("action: ", action)
            # print("new state: ", new_state)
            # print("reward: ", reward, "\n")

            # sarsamax / q-learning with temporal differences
            q[state][action] = q[state][action] + \
                lr * (reward + gamma * np.max(q[new_state]) - q[state][action])

            if terminated or truncated:
                break

            # we're now in the new state
            state = new_state

    # save q table to file to use in testing
    np.save("qtable.npy", q)

    return q

def evaluate(env,
             max_steps,
             n_eval_episodes,
             qtable):
    """Let's evaluate our Q table"""

    episode_rewards = []
    for episode in tqdm.tqdm(range(n_eval_episodes)):

        # no seed to get a different trajectory every time
        state, info = env.reset()

        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0 # total reward per episode

        for step in range(max_steps):

            action = greedy_policy(qtable, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    print("episode_rewards: ", episode_rewards)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

if __name__ == "__main__":

    #
    # Training
    #

    env = gym.make(env_id, is_slippery=False, render_mode=None)
    observation, info = env.reset(seed=42)

    state_space_size = env.observation_space.n
    print("There are ", state_space_size, " possible states")
    action_space_size = env.action_space.n
    print("There are ", action_space_size, " possible actions\n")

    qtable = initialize_q_table(state_space_size, action_space_size)

    qtable = train(
        env,
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        max_steps,
        qtable)

    #
    # Testing
    #

    env = gym.make(env_id, is_slippery=False, render_mode="human")
    mean_reward, std_reward = evaluate(env, max_steps, n_eval_episodes, qtable)

    print("mean reward: ", mean_reward)
    print("std reward: ", std_reward)

    env.close()
