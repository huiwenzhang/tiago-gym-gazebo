#!/usr/bin/env python
import argparse
from baselines import bench, logger
import gym_gazebo_ros

def train(env_id, num_timesteps, seed, policy):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy, CnnPolicy, LstmPolicy, LnLstmPolicy, CustomLstmPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    import multiprocessing
    import sys,os

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    def make_assembler_env(env_id, num_env, seed):
        # if wrapper_kwargs is None: wrapper_kwargs = {}
        def make_env(rank): # pylint: disable=C0111
            def _thunk():
                env = gym.make(env_id)
                env.seed(seed + rank)
                env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                return env
            return _thunk
        set_global_seeds(seed)
        return SubprocVecEnv([make_env(i) for i in range(num_env)])

    env = VecFrameStack(make_assembler_env(env_id, 8, seed), 4)
    policy = {'mlp': MlpPolicy, 'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'customlstm': CustomLstmPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))



    # env = DummyVecEnv([make_env])
    # env = VecNormalize(env)

    # set_global_seeds(seed)
    # ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
    #     lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
    #     ent_coef=0.0,
    #     lr=3e-4,
    #     cliprange=0.2,
    #     total_timesteps=num_timesteps)


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
    print('This example does not work correctly at present, please don not use it!')
    main()

