#!/usr/bin/env python3
import time
import gymnasium as gym
import numpy as np

# this script provides a skeleton for implementing q-learning
# the agent moves randomly in an environment and a more
# intelligent policy can be implemented, e.g. using q-learning

# Initialize the environment
env = gym.make("FrozenLake-v1", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
print("observation space", env.observation_space)

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions\n")

for _ in range(1000):

    # insert your policy here
    # ...

    # but for now sample a random action
    action = env.action_space.sample()

    # pause
    time.sleep(2)

    # step (transition) through the environment with the action
    observation, reward, terminated, truncated, info = env.step(action)

    print("action: ", action)
    print("observation: ", observation)
    print("reward: ", reward)
    print()

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        print("terminated... starting a new episode\n")
        observation, info = env.reset()

env.close()
