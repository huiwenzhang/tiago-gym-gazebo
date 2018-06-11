import argparse
import logging
import sys

# import deep reinforcement learning module
import baselines
import tensorflow as tf

import gym
from gym import wrappers

import gym_gazebo_ros

env = gym.make('TiagoReach-v0')
ob = env.reset()
print('finish reset')
env.seed(0)
print('finish seed')

# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
outdir = '/tmp/random-agent-results'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

episode_count = 100
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    time_step_num = 0
    while True:
        # sample action from action space. FIXME: make sample depend on the `ob` and `reward`, or based on the epsilon-greedy.
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        time_step_num = time_step_num + 1
        if done or (time_step_num >= 500):
            break

env.close()