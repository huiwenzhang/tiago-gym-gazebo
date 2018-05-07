#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
# from baselines.trpo_mpi import trpo_mpi

import argparse
import os
import gym
from baselines import logger, bench
import logging
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

import gym_gazebo_ros
from baselines.ppo1 import mlp_policy, pposgd_simple

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--save-model-with-prefix', type=str, help='Specify a prefix name to save the model with after every 500 iters. Note that this will generate multiple files (*.data, *.index, *.meta and checkpoint) with the same prefix', default='')
    parser.add_argument('--restore-model-from-file', type=str, help='Specify the absolute path to the model file including the file name upto .model (without the .data-00000-of-00001 suffix). make sure the *.index and the *.meta files for the model exists in the specified location as well', default='')
    args = parser.parse_args()
    return args


def train(env_id, num_timesteps, seed, save_model_with_prefix, restore_model_from_file):
    import baselines.common.tf_util as U
    # sess = U.single_threaded_session()
    # sess.__enter__()
    U.make_session(num_cpu=4).__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=15, num_hid_layers=2)
    
    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    gym.logger.setLevel(logging.WARN)

    # env = make_mujoco_env(env_id, workerseed)
    set_global_seeds(seed)
    env.seed(seed)
    # pposgd_simple.learn(env, policy_fn,
    #         max_timesteps=num_timesteps,
    #         timesteps_per_actorbatch=2048,
    #         clip_param=0.2, entcoeff=0.0,
    #         optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
    #         gamma=0.99, lam=0.95, schedule='linear',
    #     )
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.9, lam=0.95, schedule='linear', save_model_with_prefix=save_model_with_prefix,
        restore_model_from_file=restore_model_from_file)
    env.close()

def main():
    args = parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, 
            save_model_with_prefix=args.save_model_with_prefix,
            restore_model_from_file=args.restore_model_from_file)


if __name__ == '__main__':
    main()