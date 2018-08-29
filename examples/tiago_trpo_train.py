#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

import argparse
import os
import gym
from baselines import logger, bench
import logging
from baselines.common.misc_util import (
    set_global_seeds,
)

import gym_gazebo_ros


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env',
        help='environment ID',
        type=str,
        default='Reacher-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    return args


def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    # for parallel executing, each process has a rand
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=32, num_hid_layers=2)
    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(
        env, logger.get_dir() and os.path.join(
            logger.get_dir(), str(rank)))
    gym.logger.setLevel(logging.WARN)

    set_global_seeds(workerseed)
    env.seed(workerseed)
    trpo_mpi.learn(
        env,
        policy_fn,
        timesteps_per_batch=1024,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        max_timesteps=num_timesteps,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3)
    env.close()


def main():
    args = parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
