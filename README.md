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


4. Training examples

```sh
python3 examples/agents/assembler_pih_ddpg.py --env-id "Tiago-v0" --nb-rollout-steps 1000

# or set Actor network and Critic network learning rate 
python3 examples/agents/assembler_pih_ddpg.py --env-id "Tiago-v0" --nb-rollout-steps 1000 --actor-lr 0.001 --critic-lr 0.001
```

Run trpo
```
python3 examples/agents/assembler_pih_trpo.py --env "Tiago-v0" 
```


Run ppo1
```
python3 examples/agents/assembler_pih_ppo1.py --env "Tiago-v0"
```

Run ppo1 and save model 
```
python3 examples/agents/assembler_pih_ppo1.py --env "Tiago-v1" --save-model-with-prefix "/home/ros/data/DRL_Platforms/pih-logs/models/pih"
```

Run ppo1, restore model from file, and then save model  
```
python3 examples/agents/assembler_pih_ppo1.py --env "Tiago-v1"  --restore-model-from-file "/home/ros/data/DRL_Platforms/pih-logs/models/pih/openai-2018-04-18-10-49-01-988849_afterIter_489.model"  --save-model-with-prefix "/home/ros/data/DRL_Platforms/pih-logs/models/pih"
```


Run ppo2
```
python3 examples/agents/assembler_pih_reacher_ppo2.py --env "Tiago-v0"
```

Run ppo2 with our custom lstm policy:
```
python3 examples/agents/assembler_pih_reacher_ppo2.py --env "Tiago-v1" --policy "customlstm"
```

Run a2c:
```
python3 examples/agents/assembler_pih_reacher_acer.py --env "Tiago-v0" --policy "lstm" --logdir "/tmp/acer"
```

Run deploying the ppo1 model:
```
python3 examples/agents/assembler_enjoy_pih.py --env "Tiago-v1" --model-file "/home/ros/data/DRL_Platforms/pih-logs/models/pih/openai-2018-04-18-10-49-01-988849_afterIter_489.model"
```


**NOTE**: If you want to test it in jupyter notebook, you should follow this steps, export the `PYTHONPATH` environment firstly in jupyter notebook cell! 

```py
import sys
sys.path.append('/path/to/gym-gazebo-ros')
sys.path.append('/path/to/external_modules/baselines')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# set ros environment
import os 
os.environ['ROS_MASTER_URI'] = 'http://10.2.2.11:11311'
os.environ['ROS_IP'] = '10.2.2.16'

# set ros environment, or will get ros init_node or ros resource not found error.
os.environ['ROS_ROOT'] = '/opt/ros/kinetic/share/ros'
os.environ['ROS_PACKAGE_PATH'] = '/opt/ros/kinetic/share'
```

Then you can test the examples in jupyter.

# Developer Guide

Currently, the design of this project is raw.

- Access some variables defined in `env` using `unwrapped` 
```py
# access env.time_step_index
env.unwrapped.time_step_index

# access env.state
env.unwrapped.state
```
