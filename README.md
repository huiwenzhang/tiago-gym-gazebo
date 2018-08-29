# Tiago-gym-gazebo
A TIAGo environment used for manipulation tasks learning based on ros and openai gym

# Usage

1. Start ROS launch file and make sure necessary nodes, topics, servers are runing. We can start by runing code:` roslaunch tiago_gazebo tiago_gazebo.launch robot:=titanium world:=empty public_sim:=true`.

2. Test related env in gym

3. test env example
```py
import gym
import gym_gazebo_ros
env = gym.make('Tiago-v0')
env.reset()
env.render()
```
