import gym

# test classical control
env = gym.make('CartPole-v0')
env.reset()
for _ in range(300):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
print("Control env installed successfully")

# test atari
env = gym.make('Breakout-v0')
env.reset()
for _ in range(300):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
print("Atari env installed successfully")

# test MuJoCo
# env = gym.make('Hopper-v0')
# env.reset()
# for _ in range(100):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()
