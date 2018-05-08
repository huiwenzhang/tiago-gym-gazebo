import gym_gazebo_ros
import gym
env = gym.make('Tiago-v0')
env.reset()
env.reach_to_ball()
# for i in range(1000):
#     action = env.action_space.sample()
#     s, r, done, _ = env.step(action)
#     if done:
#         env.reset()
# env.render()