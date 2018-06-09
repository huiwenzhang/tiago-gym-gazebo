#!/usr/bin/env python3
import argparse
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
import gym
import gym_gazebo_ros

from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--logdir', help ='Directory for logging')
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    return args

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    env = gym.make(env_id)
    set_global_seeds(seed)
    env.seed(seed)
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    args = parse_args()
    logger.configure(args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)

if __name__ == '__main__':
    main()