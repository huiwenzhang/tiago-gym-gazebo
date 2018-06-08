import gym_gazebo_ros
import gym
# env = gym.make('Tiago-v0')
env = gym.make('TiagoReach-v2')
# env = gym.make('TiagoPick-v0')
env = env.unwrapped
env.reset()
# env.reach_to_point()
for i in range(1000):
    action = env.action_space.sample()
    action = [0.02*x - 0.01 for x in action]
    s, r, done, _ = env.step(action)
    if done:
        env.reset()
# env.render()