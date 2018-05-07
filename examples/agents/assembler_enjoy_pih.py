import gym
import gym_gazebo_ros
import argparse
from baselines import deepq


# import model
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.ppo1.mlp_policy import MlpPolicy


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Assembler-v1')
    parser.add_argument('--model-file', type=str, help='Specify the absolute path to the model file including the file name upto .model (without the .data-00000-of-00001 suffix). make sure the *.index and the *.meta files for the model exists in the specified location as well', default='')
    args = parser.parse_args()
    return args


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=15, num_hid_layers=2)

def load_model(env, model_file):
    U.make_session(num_cpu=4).__enter__()

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy

    saver = tf.train.Saver()
    # with U.get_session() as sess:
    saver.restore(U.get_session(), model_file)

    return pi


def main():
    args = parse_args()
    env = gym.make(args.env)
    act_policy = load_model(env,args.model_file)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            ac, vpred = act_policy.act(False, obs)
            obs, rew, done, _ = env.step(ac)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()