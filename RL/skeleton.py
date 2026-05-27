#!/usr/bin/env python3
import time
import gymnasium as gym
import numpy as np

# this script provides the skeleton for q learning
# the agent moves randomly in an environment
# ready to fill in the details of q-learning implementation

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

for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # pause
    time.sleep(2)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    print("action: ", action)
    print("observation: ", observation)
    print("reward: ", reward)
    print()

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
