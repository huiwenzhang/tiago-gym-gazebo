# tiago-gym-gazebo
A TIAGo environment used for manipulation tasks learning based on ros and openai gym

# Usage

1. add current directory path to `PYTHONPATH` environment

```sh
export PYTHONPATH=$PYTHONPATH:/path/to/gym-gazebo-ros:/path/to/catkin_assembler_ros_python
```

**NOTE:** we can firstly `source` the `catkin_assembler_ros` ros workspace, and then add the `gym-gazebo-ros` path to `PYTHONPATH`, like
```sh
# cd /path/to/PyKDL_and_kdl_utils_catkin_ws
cd /path/to/catkin_assembler_ros
# If you need compile pykdl with python3, run the following command
# catkin build  -DPYTHON_VERSION=3.5 -DPYTHON_EXECUTABLE=/usr/bin/python3.5
source devel/setup.bash

# and we can add the pykdl_utils python pacakge path to this

export PYTHONPATH=$PYTHONPATH:/path/to/gym-gazebo-ros
```

~~And we also consider using [ros/kdl_parser](https://github.com/ros/kdl_parser) to replace same part of [kdl_utils](https://github.com/jacknlliu/hrl-kdl.git), i.e. `kdl_parser` will be a submodule of `kdl_utils`.~~



2. add `baselines` path to `PYTHONPATH`
```
export PYTHONPATH=/path/to/external_modules/baselines:$PYTHONPATH
```

In summary,
```
export PYTHONPATH=/home/ros/data/DRL_Platforms/external_modules/baselines:/home/ros/data/DRL_Platforms/gym-gazebo-ros:$PYTHONPATH
```


3. test env unit
```py
import gym
import gym_gazebo_ros
env = gym.make('Assembler-v0')
env.reset()
env.render()
```


4. test examples

```sh
python3 examples/agents/assembler_pih_ddpg.py --env-id "Assembler-v0" --nb-rollout-steps 1000

# or set Actor network and Critic network learning rate 
python3 examples/agents/assembler_pih_ddpg.py --env-id "Assembler-v0" --nb-rollout-steps 1000 --actor-lr 0.001 --critic-lr 0.001
```

Run trpo
```
python3 examples/agents/assembler_pih_trpo.py --env "Assembler-v0" 
```


Run ppo1
```
python3 examples/agents/assembler_pih_ppo1.py --env "Assembler-v0"
```

Run ppo1 and save model 
```
python3 examples/agents/assembler_pih_ppo1.py --env "Assembler-v1" --save-model-with-prefix "/home/ros/data/DRL_Platforms/pih-logs/models/pih"
```

Run ppo1, restore model from file, and then save model  
```
python3 examples/agents/assembler_pih_ppo1.py --env "Assembler-v1"  --restore-model-from-file "/home/ros/data/DRL_Platforms/pih-logs/models/pih/openai-2018-04-18-10-49-01-988849_afterIter_489.model"  --save-model-with-prefix "/home/ros/data/DRL_Platforms/pih-logs/models/pih"
```


Run ppo2
```
python3 examples/agents/assembler_pih_reacher_ppo2.py --env "Assembler-v0"
```

Run ppo2 with our custom lstm policy:
```
python3 examples/agents/assembler_pih_reacher_ppo2.py --env "Assembler-v1" --policy "customlstm"
```

Run a2c:
```
python3 examples/agents/assembler_pih_reacher_acer.py --env "Assembler-v0" --policy "lstm" --logdir "/tmp/acer"
```

Run deploying the ppo1 model:
```
python3 examples/agents/assembler_enjoy_pih.py --env "Assembler-v1" --model-file "/home/ros/data/DRL_Platforms/pih-logs/models/pih/openai-2018-04-18-10-49-01-988849_afterIter_489.model"
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
