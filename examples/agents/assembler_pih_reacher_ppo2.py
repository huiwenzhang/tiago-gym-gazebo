#!/usr/bin/env python
import argparse
from baselines import bench, logger
import gym_gazebo_ros

def train(env_id, num_timesteps, seed, policy):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy, CnnPolicy, LstmPolicy, LnLstmPolicy, CustomLstmPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        # print('1 in make env, the env.num_envs is {}'.format(env.num_envs))
        env = bench.Monitor(env, logger.get_dir())
        print('bench monitor env is: {}'.format(env))
        # print('2 in make env, the env.num_envs is {}'.format(env.num_envs))
        return env

    print('make env is: {}'.format([make_env]))
    env = DummyVecEnv([make_env])
    print('dummy vec env, the env.num_envs is {}'.format(env.num_envs))
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = {'mlp': MlpPolicy, 'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'customlstm': CustomLstmPolicy}[policy]
    # as default, we set nminibatches=32
    if policy == 'customlstm':
        nminibatches = 1
    else:
        nminibatches = 32

    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=nminibatches,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'cnn', 'lstm', 'lnlstm', 'customlstm'], default='mlp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy)


if __name__ == '__main__':
    main()

