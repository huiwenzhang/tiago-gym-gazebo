import gym
import rospy
import roslaunch
from geometry_msgs.msg import Twist, WrenchStamped
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray


import time
from threading import Timer, Lock
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

# deep learning modules. reference to gym/envs/parameter_tuning/train_deep_cnn.py
# from keras.datasets import cifar10, mnist, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
# from keras.regularizers import WeightRegularizer
# using tensorflow as backend of keras
from keras import backend as K

from itertools import cycle


# ros robot related
from gazebo_msgs.srv import GetModelState, GetLinkState, SetModelConfiguration, DeleteModel, \
    SpawnModel, SetModelState

from gazebo_msgs.msg import ModelState, ContactState, ContactsState


from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing UR joint angles.

from urdf_parser_py.urdf import URDF

# from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

## TODO: we should create a command interface(based on ros middle-ware, topic, service or action) 
## between learning module and robot manipulation module.
## But for simplify the program, we leave this work to do. Never do the robot things here! Keep a very pure thing
## about learning here!
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *

from . import ros_general_utils as ros_utils # custom user defined ros utils

# math module
import math
import transforms3d as tf3d
from numpy.linalg import inv, norm

class AssemblerPiHEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_node=True):
        """Initialized Assembler robot Env
        init_node:   Whether or not to initialize a new ROS node.
        """
        # (needed by gazebo_ros) ros packages out of this source, 
        # we should make sure that we have launch the simulation using external launch file.

        # (needed by gazebo_ros)Setup the main node, if we call this from outside, 
        # we may set init_node=False to disable generate a new node.
        if init_node:
            # if not init a ros node, we can use the ros topic related method, such as rospy.wait_for_message()
            rospy.init_node('gym_ros_node')
            print('initialize ros node')

        self.__joint_state_sub = rospy.Subscriber("/joint_states", JointState,
                                                  self.__joint_state_cb, queue_size=5)

        rospy.wait_for_service("/gazebo/reset_world", 10.0)
        self.__reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        rospy.wait_for_service("/gazebo/get_model_state", 10.0)
        self.__get_pose_srv = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState)

        rospy.wait_for_service("/gazebo/get_link_state", 10.0)
        self.__get_link_pose_srv = rospy.ServiceProxy(
            "/gazebo/get_link_state", GetLinkState
        )
        rospy.wait_for_service("/gazebo/pause_physics")
        self.__pause_physics = rospy.ServiceProxy(
            "/gazebo/pause_physics", Empty)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.__unpause_physics = rospy.ServiceProxy(
            "/gazebo/unpause_physics", Empty)
        
        rospy.wait_for_service("/controller_manager/switch_controller")
        self.__switch_ctrl = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)

        rospy.wait_for_service("/gazebo/set_model_configuration")
        self.__set_model = rospy.ServiceProxy(
            "/gazebo/set_model_configuration", SetModelConfiguration)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.__set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)


        rospy.wait_for_service("/gazebo/delete_model")
        self.__delete_model = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel)
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.__spawn_model = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel)

        rospy.Subscriber("/robot_fixed_box_bumper", ContactsState, self._robot_fixed_box_contact_callback)
        rospy.Subscriber("/filtered/ft_sensor_data", WrenchStamped, self._ee_baseplate_contact_callback)
        rospy.Subscriber("/filtered/ft_sensor_total_data", WrenchStamped, self._ee_sum_contact_callback)
        self.ee_baseplate_ft=[0.0]*6
        self.ee_sum_contact_ft=[0.0]*6


        self.lock = Lock()
        self.contact_flag_released = True
        self.contact_flag = False


        # ros velocity controller
        self.pub_vel = rospy.Publisher("joint_group_velocity_controller/command", Float64MultiArray, queue_size=1)

        print("finish setup ros services")

        # TODO: we will replace the control api to a non-lib method using ros middle-ware
        self.__arm_joint_traj_client = actionlib.SimpleActionClient('/arm_gazebo_controller/follow_joint_trajectory',
                                                                     FollowJointTrajectoryAction)
        # self.__hand_joint_traj_client = actionlib.SimpleActionClient('/gripper_gazebo_controller/follow_joint_trajectory', 
        #                                                             FollowJointTrajectoryAction)

        self.__current_object = '50-cylinder-peg-gap-3mm'

        # (needed by gym) define action space, observation space, reward range
        # using URDF setup action_space and observation space, reward.
        self.robot = URDF.from_parameter_server()
        self.joint_names_order = [self.robot.joints[i].name for i in range(6)]
        self.global_joint_names_order = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        assert self.joint_names_order == self.global_joint_names_order

        # here we know is:
        # ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        # we follow this order in joint velocity controller and joint state publisher
        print("joint names order from URDF is: {}".format(self.joint_names_order))

        self.joint_limits_lower = [self.robot.joints[i].limit.lower for i in range(6) ]
        self.joint_limits_upper = [self.robot.joints[i].limit.upper for i in range(6) ]

        joint_effort = [self.robot.joints[i].limit.effort for i in range(6)]

        self.joint_vel_limits_upper=[self.robot.joints[i].limit.velocity for i in range(6)]
        self.joint_vel_limits_lower=[-self.robot.joints[i].limit.velocity for i in range(6)]

        self.ee_distance_lower=[-np.inf]*3 + [-1.0]*4 # position [x,y,z] and quanternion (w,x,y,z)
        self.ee_distance_upper=[np.inf]*3 + [1.0] * 4

        self.ee_vel_lower= [-np.inf]*6
        self.ee_vel_upper= [np.inf]*6

        self.low_state = np.array(self.ee_distance_lower)
        self.high_state = np.array(self.ee_distance_upper)

        self.control_period = 0.01 # unit is: seconds
        # self.action_space = spaces.Box(np.array(self.joint_limits_lower), np.array(self.joint_limits_upper))
        # 10.0 is used to make the action step more smaller
        self.action_lower = np.array(self.joint_vel_limits_lower)
        self.action_upper = np.array(self.joint_vel_limits_upper)
        self.action_space = spaces.Box(self.action_lower, self.action_upper)


        self.force_sensor_lower = [-np.inf]*6
        self.force_sensor_upper = [np.inf]*6
        self.low_full_state = np.concatenate((self.low_state, self.force_sensor_lower))
        self.high_full_state = np.concatenate((self.high_state, self.force_sensor_upper))
        # self.observation_space = spaces.Box(np.array(self.low_state), np.array(self.high_state))
        self.observation_space = spaces.Box(np.array(self.low_full_state), np.array(self.high_full_state))
        self.reward_range = (-np.inf, np.inf)

        self.state = None
        self.__last_joint_state = None

        # define target pose
        # Rotation = tf3d.euler.euler2mat(-1.649357,-0.032365, -1.497718, 'sxyz')
        # Rotation = tf3d.euler.euler2mat(-1.458,-0.1353, -1.906, 'sxyz')
        self.home_arm_joints_position = [-0.53, -0.44, 1.13, -2.32, -1.57, -0.75]
        
        self.hand_joint_names_order = ['bh_j11_joint', 'bh_j12_joint', 'bh_j13_joint',
               'bh_j21_joint', 'bh_j22_joint', 'bh_j23_joint', 'bh_j32_joint', 'bh_j33_joint']
        self.home_hand_joints_position = [1.567913, 1.691401, 0.020330, 1.567875, 1.691463, 0.022318, 2.422133, 0.822189]

        Rotation = tf3d.euler.euler2mat(-1.570796, 0.0, -3.141593, 'sxyz')
        # Translation = [0.721576, -0.116532, 1.308585]
        # Translation = [0.721576, -0.16, 1.308585]
        # Translation = [0.735555,-0.306439, 1.09400]
        Translation = [0.735555,-0.306439, 1.15400]
        self.ee_target_pose = tf3d.affines.compose(Translation, Rotation, np.ones(3))

        self.tolerance = 1e-4


        self.safety_bound_points = np.array([[0.584671, -0.474691, 1.092184], [0.883217, -0.474691, 1.092184],
                               [0.883217, -0.113535, 1.092184], [0.584671, -0.113535, 1.092184],
                               [0.584671, -0.474691, 1.484668], [0.883217, -0.474691, 1.484668],
                               [0.883217, -0.113535, 1.484668], [0.584671, -0.113535, 1.484668]])

        # define the number of time step for every step know the time, then it can compute something
        # depend on the time
        self.time_step_index = 0

        # self.max_episode_steps = 10000 # only used for some algorithm, thi is complicated?


        # (needed by gym) seed and reset
        self._seed()
        self._reset()

        print("finish setup assembler env.")

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _robot_fixed_box_contact_callback(self, bumperstate):
        self.lock.acquire()
        if len(bumperstate.states) >= 1:
            self.contact_flag = True
            self.contact_flag_released = False
        self.lock.release()
    
    def _ee_baseplate_contact_callback(self, ee_baseplate_contact):
        self.ee_baseplate_ft = [ee_baseplate_contact.wrench.force.x, ee_baseplate_contact.wrench.force.y, ee_baseplate_contact.wrench.force.z,
                                ee_baseplate_contact.wrench.torque.x, ee_baseplate_contact.wrench.torque.y, 
                                ee_baseplate_contact.wrench.torque.z]


    def _ee_sum_contact_callback(self, ee_sum_contact):
        self.ee_sum_contact_ft = [ee_sum_contact.wrench.force.x, ee_sum_contact.wrench.force.y, ee_sum_contact.wrench.force.z,
                                    ee_sum_contact.wrench.torque.x, ee_sum_contact.wrench.torque.y, ee_sum_contact.wrench.torque.z]

    def _get_ee_baseplate_ft(self):
        return self.ee_baseplate_ft

    def _get_ee_sum_ft(self):
        return self.ee_sum_contact_ft

    def _get_ee_all_ft(self):
        return self.ee_baseplate_ft, self.ee_all_contact_ft


# discretize_observation() is not need for this env
    def discretize_observation(self, data, new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges) / new_ranges
        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if data.ranges[i] == float('Inf'):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges, done

    def _step(self, action):
        """
        Perform some action in the environment

        action: depend on the action space setting

        Return:  state, reward, and done status in this function

        Many times, we need make a step from fit a state using deep learning model by keras or RL by baselines
        """
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # send action to joint trajctory
        # self.__arm_joint_traj_client.wait_for_server()
        # rospy.loginfo('connected to robot arm controller server')

        # g = FollowJointTrajectoryGoal()
        # g.trajectory = JointTrajectory()
        # g.trajectory.joint_names = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
        # 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        # slow_duration =0.1

        joint_data = None
        # while joint_data is None:
        #     joint_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
        joint_data = self.__last_joint_state

        self.lock.acquire()
        contact_flag = self.contact_flag
        contact_flag_released = self.contact_flag_released
        self.lock.release()

        if contact_flag_released is False:
            if contact_flag is True:
                state, ee_absolute_translation, joint_position, joint_vel, _ = self._get_obs()
                # stop move
                rospy.loginfo("==== stop robot for contact")

                self.zero_velocity_stop()

                # pause the simulation
                rospy.loginfo("==== pause physics for contact")
                rospy.wait_for_service('/gazebo/pause_physics')
                try:
                    self.__pause_physics()
                except (rospy.ServiceException) as exc:
                    print("/gazebo/pause_physics service call failed:" + str(exc))
        else:
            try:
                # TODO: actually our action should consider the robot joint limit (including velocity limit)
                # TODO: add action prediction and corresponding terminate condition prediction before take excution
                # TODO: limit action (position translocation), keep every step have a very small moving. 
                # we use joint position increment to send to robot
                print("============new action is : {}".format(action))
                # positions_point = np.clip(action + np.array(joint_data.position),self.joint_limits_lower, self.joint_limits_upper)
                # Q_target = positions_point.tolist()
                # g.trajectory.points = [JointTrajectoryPoint(positions=Q_target, velocities=[0]*6, time_from_start=rospy.Duration(self.control_period))]
                # self.__arm_joint_traj_client.send_goal(g)

                # for safety action, perhaps we need a basic compliance control
                # if max(np.fabs(self._get_ee_sum_ft())) > 300.0:
                #     self.zero_velocity_stop()
                # else:
                vel_msg = Float64MultiArray()

                vel_msg.data = np.clip(action, self.action_lower, self.action_upper).tolist()

                self.pub_vel.publish(vel_msg)

                rospy.loginfo('send vel to robot arm')
                # bug? wait for result blocking!
                # self.__arm_joint_traj_client.wait_for_result()
                time.sleep(self.control_period)
                rospy.loginfo('vel of robot arm past a period')
                # get joint data. 
                state, ee_absolute_translation, joint_position, joint_vel, _ = self._get_obs()
            except KeyboardInterrupt:
                self.__arm_joint_traj_client.cancel_goal()


        # TODO: add get the filtered force sensor data

        # # get joint data. (comment for new test)
        # state, ee_absolute_translation = self._get_obs()

        # actually I think we don't need pause physics! 
        # because we can't pause the continous motion in a episode
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     self.__pause_physics()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")

        self.state = state
        current_jacobian = self._get_jacobian(joint_position)

        # (needed by gym )done is the terminate condition. We should
        # return this value by considering the terminal condition.
        # TODO:  timestep stop set in max_episode_steps when register this env. 
        # Another is using --nb-rollout-steps in running this env example?
        done = self.is_task_done(state, ee_absolute_translation, joint_vel)

        # self.time_step_index = self.time_step_index % self.spec.timestep_limit

        # TODO: add weight coss for different timestep.
        # weight_coss = (self.time_step_index+1)/self.spec.timestep_limit
        weight_coss = 1.0

        # self.time_step_index += 1

        # rospy.loginfo('episode steps is {}, time step index is {}'.format(self.spec.timestep_limit, self.time_step_index))

        # TODO: add jacobian cost as part of reward, like GPS, which can avoid the robot explore the sigularity position.
        #  Also can add joint torque as part of reward, like GPS. 
        # reward = max(0.0, 1.5 - 0.01*norm(np.array(end_pose_dist))**2)
        end_pose_dist = self._extract_end_pose_dist_from_state(state)
        # distance = np.sqrt( np.sum(np.array(end_pose_dist[:3])**2) + ros_utils.distance_of_quaternion(end_pose_dist[3:7])**2 )
        # We weigh the difference in position to avoid that `pose` (in meters) is completely
        # dominated by `quaternion` (in radians).
        # distance = 10 * norm(np.array(end_pose_dist[:3])) + ros_utils.distance_of_quaternion(end_pose_dist[3:7])
        end_translation_dist = norm(np.array(end_pose_dist[:3]))
        end_rotation_dist = ros_utils.distance_of_quaternion(end_pose_dist[3:7])
        distance = 3.0 * np.exp(-2 * end_translation_dist) - end_rotation_dist/np.pi
        # TODO: we should add safety bounds information to reward, not only terminate condition,like some paper!
        force_data_term =  np.array(self._extract_ft_from_state(state))
        force_penalty_term = np.exp(-norm(force_data_term)/1800.0)
        manipulation_term = norm(current_jacobian, ord=-2)/norm(current_jacobian, ord=2)
        action_penalty_term = - np.square(action).sum() / np.square(self.joint_vel_limits_upper).sum()

        # TODO: we weight the force data term.
        # reward = - distance - 0.01 * norm(force_data_term)
        reward = distance + force_penalty_term + action_penalty_term # candidate
        # reward = - distance - 1e-7 * norm(force_data_term) - action_penalty_term
        # reward = - 100 * norm(np.array(end_pose_dist[:3])) - action_penalty_term

        # using sparse reward
        # reward = ((distance < self.tolerance) and (norm(force_data_term)<10) and (action_penalty_term <0.1)).astype(np.float32) 

        print("===================reward is: {}==============================".format(reward))
        print("===================distance is : {}===========================".format(distance))

        # (needed by gym) we should return the state(or observation from state(function of state)), reward, and done status.
        # If the task completed, such as distance to target is d > = 0.001,
        # then return done= True, else return done = False. done depend on the terminal conditions partly.
        # NOTE: `reward` always depend on done and action, or state, so it always calculated finally.

        # self.time_step_index = self.time_step_index + 1
        # print('in current step time_step_index is: {}, done value is: {}'.format(self.time_step_index, done))
        return np.array(self.state), reward, done, {}


    def zero_velocity_stop(self):
        vel_msg = Float64MultiArray()

        vel_msg.data = [0.0] *6

        self.pub_vel.publish(vel_msg)


    def _get_obs(self, immediately=False):
        joint_data = None
        if immediately is True:
            while joint_data is None:
                try:
                    rospy.loginfo('try to receive robot joint states...')
                    joint_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
                except KeyboardInterrupt:
                    break
        else:
            joint_data = self.__last_joint_state

        # get joint position and velocity
        joint_position = list(joint_data.position)
        joint_vel = list(joint_data.velocity)
        # get end-effector position and distance to target and end-effector velocity
        end_pose_dist, end_pose_vel, ee_absolute_translation = self.get_ee_state()

        # get wrist force sensor data
        # force_data_msg = None
        # while force_data_msg is None:
        #     rospy.loginfo('try to receive force sensor data...')
        #     force_data_msg = rospy.wait_for_message('/filtered/ft_sensor_data', WrenchStamped, timeout=5)
        
        force_data = self._get_ee_baseplate_ft()

        state =  end_pose_dist + force_data

        return state, ee_absolute_translation, joint_position, joint_vel, end_pose_vel

    def _extract_end_pose_dist_from_state(self, state):
        return state[0:7]

    def _extract_ft_from_state(self, state):
        return state[7:13]


    def _reset(self):
                # we should stop our controllers firstly or send the initial joint angles to robot 
        # using joint trajectory controller. Or the robot will back to its current position after unpause the simulation

        # reset arm position
        rospy.loginfo('start to reset environment')

        self._reset_world_origin()

        rospy.loginfo('fininshed reset environment')

        # read data to observation

        # update `state`
        # state = self.discretize_observation(data, 5)
        state,_, _, _,_= self._get_obs(immediately=True)
        self.state = state

        # for debug
        print('in reset. current time step index is {}'.format(self.time_step_index))

        # (needed by gym) return the initial observation or state
        return np.array(state)


    def _reset_world_origin(self):

        # stop controllers
        self.__switch_ctrl.call(stop_controllers=["arm_gazebo_controller", "joint_group_velocity_controller"],
                start_controllers=[],
                    strictness=SwitchControllerRequest.BEST_EFFORT)

        # pause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.__pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))
        
        # set robot model configuration
        joint_names = self.joint_names_order
        joint_positions = self.home_arm_joints_position

        self.__set_model.call(model_name="robot", urdf_param_name="robot_description",
                              joint_names=joint_names, joint_positions = joint_positions)

        self.__unpause_physics.call()

        time.sleep(0.1)

        timer = Timer(0.0, self.__switch_controllers)
        timer.start()


    def __switch_controllers(self):
        rospy.loginfo("STARTING CONTROLLERS")
        self.__switch_ctrl.call(start_controllers=["joint_group_velocity_controller"], 
                        stop_controllers=["arm_gazebo_controller"], strictness=SwitchControllerRequest.BEST_EFFORT)


    def __start_ctrl(self):
        rospy.loginfo("STARTING CONTROLLERS")
        self.__switch_ctrl.call(start_controllers=["arm_gazebo_controller", "joint_state_controller"], 
                        stop_controllers=[], strictness=1)


    def _render(self, mode='human', close=False):
        """Send state to plot in the window. This is not needed by gazebo env.

        Args:
            mode (str): render mode. 'human' for no render on the window
            close (bool): True for close the window, False for keep open
        
        Returns:
            return_type: return nothig.

        Raises:
            AttributeError: xxx
            ValueError: xxx
        """
        if close:
            return
        # if not close, we print the process data, such as steps of epoch (epoch index)
        # print(">> Step ", self.epoch_idx, "best validation:", self.best_val)


    def _reset_world_scene(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.__pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))

        # this will reset the simulation time, and this will affect the ros controllers
        rospy.loginfo("reset world")
        self.__reset_world()

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))


    def _reset_robot_arm_pose_first_stage(self):
        """Reset the pose of robot arm in gazebo to home pose.
        
        We will reset the robot arm pose.
        """
        
        # reset robot arm
        # using robot joint trajectory controller to send robot joint angles
        # we should not reset the hand, because re-grasp is not easy

        # switch controller
        # we should unpause the physics before switching the controller
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        self.__switch_ctrl.call(start_controllers=["arm_gazebo_controller"],
                        stop_controllers=["joint_group_velocity_controller"],
                            strictness=SwitchControllerRequest.BEST_EFFORT)

        # reset robot arm to lift pose using joint trajectory controller
        rospy.loginfo('wait for robot arm joint trajectory controller server')
        self.__arm_joint_traj_client.wait_for_server()
        rospy.loginfo('connected to robot arm controller server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal 
        self.__arm_joint_traj_client.cancel_goal()

        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()

        g.trajectory.joint_names = self.joint_names_order
        slow_duration =1.0
        try:
            # Q_target = [1.057810,-0.683443,-0.532145,-1.944877,-1.570304,-2.100711]
            # Q_target = [1.38,-0.75,-0.38,-2.32,-1.7,-0.06]
            Q_target = self.home_arm_joints_position
            g.trajectory.points = [JointTrajectoryPoint(positions=Q_target, velocities=[0]*6, time_from_start=rospy.Duration(slow_duration))]
            self.__arm_joint_traj_client.send_goal(g)
            rospy.loginfo('reset first stage send goal to robot arm')
            # bug? wait for result blocking!
            # self.__arm_joint_traj_client.wait_for_result()
            time.sleep(slow_duration*2)
            rospy.loginfo('reset first stage goal of robot arm completed')
        except KeyboardInterrupt:
            self.__arm_joint_traj_client.cancel_goal()


    def _reset_robot_arm_pose(self):
        """Reset the pose of robot arm in gazebo to home pose.
        
        We will reset the robot arm pose.
        """
        
        # reset robot arm
        # using robot joint trajectory controller to send robot joint angles
        # we should not reset the hand, because re-grasp is not easy

        # switch controller
        # we should unpause the physics before switching the controller
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        self.__switch_ctrl.call(start_controllers=["arm_gazebo_controller"],
                        stop_controllers=["joint_group_velocity_controller"],
                            strictness=SwitchControllerRequest.BEST_EFFORT)

        # reset robot arm to lift pose using joint trajecotry controller
        rospy.loginfo('wait for robot arm joint trajectory controller server')
        self.__arm_joint_traj_client.wait_for_server()
        rospy.loginfo('connected to robot arm controller server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal 
        self.__arm_joint_traj_client.cancel_goal()

        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()

        g.trajectory.joint_names = self.joint_names_order
        slow_duration =1.0
        try:
            # Q_target = [1.057810,-0.683443,-0.532145,-1.944877,-1.570304,-2.100711]
            # Q_target = [1.38,-0.75,-0.38,-2.32,-1.7,-0.06]
            Q_target = self.home_arm_joints_position
            Q_target = np.clip(Q_target + self.np_random.uniform(-0.2,0.2,(6,)), self.joint_limits_lower, self.joint_limits_upper).tolist()
            g.trajectory.points = [JointTrajectoryPoint(positions=Q_target, velocities=[0]*6, time_from_start=rospy.Duration(slow_duration))]
            self.__arm_joint_traj_client.send_goal(g)
            rospy.loginfo('send goal to robot arm')
            # bug? wait for result blocking!
            # self.__arm_joint_traj_client.wait_for_result()
            time.sleep(slow_duration*2)
            rospy.loginfo('goal of robot arm completed')
        except KeyboardInterrupt:
            self.__arm_joint_traj_client.cancel_goal()


    def _reset_hand_pose(self):
        """Reset hand pose in gazebo.
        
        Reset the hand pose in gazebo.
        """
        # rospy.loginfo('wait for hand joint trajectory controller server')
        # self.__hand_joint_traj_client.wait_for_server()
        # rospy.loginfo('Connected to hand controller server')

        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = self.hand_joint_names_order
        slow_duration = 0.1
        try:
            Q_target = self.home_hand_joints_position
            g.trajectory.points = [JointTrajectoryPoint(positions=Q_target, velocities=[0.1]*8, time_from_start=rospy.Duration(slow_duration))]
            self.__hand_joint_traj_client.send_goal(g)
            rospy.loginfo('send goal to robot hand')
            # bug? wait for rest blocking
            # self.__hand_joint_traj_client.wait_for_result()
            time.sleep(slow_duration*2)
            rospy.loginfo('goal of robot hand completed')
        except KeyboardInterrupt:
            self.__hand_joint_traj_client.cancel_goal()


    def _set_object_model_pose(self, model_name, model_pose, model_velocity=[0.0]*6, reference_frame='world'):
        """Set Ojbect model pose for the four object.
        
        Four models ('50-cylinder-peg-gap-3mm', 'triangle-peg-with-3mm-gap', '50-rectangle-3mm',
        'baseplate') can be selected.

        Arguments:
            model_name {str} -- model name of object in the scene
            model_pose {list} -- mode pose list, first three number for position, last three number for orientation
            model_velocity {list} -- mode velocity list, first three number for linear, last three for angular
        """
        model_state_msg = ModelState()
        model_state_msg.model_name = model_name
        model_state_msg.pose.position.x = model_pose[0]
        model_state_msg.pose.position.y = model_pose[1]
        model_state_msg.pose.position.z = model_pose[2]
        model_state_msg.pose.orientation.x = model_pose[3]
        model_state_msg.pose.orientation.y = model_pose[4]
        model_state_msg.pose.orientation.z = model_pose[5]
        model_state_msg.twist.linear.x = model_velocity[0]
        model_state_msg.twist.linear.y = model_velocity[1]
        model_state_msg.twist.linear.z = model_velocity[2]
        model_state_msg.twist.angular.x = model_velocity[3]
        model_state_msg.twist.angular.y = model_velocity[4]
        model_state_msg.twist.angular.z = model_velocity[5]
        model_state_msg.reference_frame = reference_frame
        self.__set_model_state(model_state_msg)


    def _reset_object_pose(self, ex_object_name):
        """Reset target object pose in gazebo.
        
        Using gazebo interface to reset object pose, but not the `ex_object_name`
        
        Arguments:
            ex_object_name (str): the name of object, which we will not reset.
        """

        # pause simulation 
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.__pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))

        # reset object. 
        # reset current target object, before this, we should **reset the world** firstly.
        model_name = self.__current_object
        if self.__current_object == '50-cylinder-peg-gap-3mm':
            rospy.loginfo('reset cylinder peg')
            # set to default grasp position is not easy, because the robot re-grasp can't only depend on the robot hand. 
            model_pose = [0.735317, -0.306257, 1.204858, -0.000388, -0.000580, -0.001478]
            self._set_object_model_pose(model_name=model_name, model_pose=model_pose)
        elif self.__current_object == 'triangle-peg-with-3mm-gap':
            model_pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            self._set_object_model_pose(model_name=model_name, model_pose=model_pose)
        elif self.__current_object == '50-rectangle-3mm':
            model_pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            self._set_object_model_pose(model_name=model_name, model_pose=model_pose)
        elif self.__current_object == 'baseplate':
            rospy.loginfo('reset baseplate peg')
            model_pose = [0.735555, -0.306439, 1.093998, 0, 0, -1.573690]
            self._set_object_model_pose(model_name=model_name, model_pose=model_pose)

        # unpause simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))


    def __joint_state_cb(self, msg):
        self.__last_joint_state = msg


    def get_ee_state(self):
        """Get end effector pose relative to the target.
        
        Return target pose relative to the target.
        """
        rospy.wait_for_service('/gazebo/get_link_state')

        try:
            end_state = self.__get_link_pose_srv.call('robot::ee_link', "world").link_state
        except (rospy.ServiceException) as exc:
            print("/gazebo/get_link_state service call failed:" + str(exc))
        
        end_pose_msg = end_state.pose
        ee_vel_msg = end_state.twist

        ###### start to extract the msg to data ######

        # translate the pose msg type to numpy.ndarray
        q = [end_pose_msg.orientation.w, end_pose_msg.orientation.x, end_pose_msg.orientation.y, end_pose_msg.orientation.z]
        Rotation = tf3d.quaternions.quat2mat(q)
        Translation = [end_pose_msg.position.x, end_pose_msg.position.y, end_pose_msg.position.z]
        end_pose_affine = tf3d.affines.compose(Translation, Rotation, np.ones(3))

        end_absolute_translation = end_pose_affine[:3, 3].reshape(-1).tolist()

        # compute the relative pose to target pose (target frame relative the current frame)
        ee_pose_dist_affine = np.dot(inv(end_pose_affine), self.ee_target_pose)

        # translate the relative pose affine matrix to list (position, quaternion)
        ee_translation = ee_pose_dist_affine[:3,3].reshape(-1).tolist()
        ee_quat = tf3d.quaternions.mat2quat(ee_pose_dist_affine[:3,:3]).tolist()
        ee_pose_dist = ee_translation + ee_quat

        # form the end-effector twist list
        ee_velocity = [ee_vel_msg.linear.x, ee_vel_msg.linear.y, ee_vel_msg.linear.z, 
                        ee_vel_msg.angular.x, ee_vel_msg.angular.y, ee_vel_msg.angular.z]

        return ee_pose_dist, ee_velocity, end_absolute_translation

    def is_task_done(self, state, end_absolute_translation, joint_vel):

        # extract end pose distance from state
        end_pose_dist =  self._extract_end_pose_dist_from_state(state)

        # TODO: add collision detection to cancel wrong/bad/negative trial!
        # TODO: add end-effector force sensor data to terminate the trial

        # detect safety bound
        is_in_bound = self._contain_point(self.safety_bound_points, end_absolute_translation)
        if is_in_bound is False:
            print("=================robot out of box=====================")
            print("=================current end absolute pose is:=========================")
            print(end_absolute_translation)
            return True

        # extract end-effector force sensor data from state
        # force_data = state[13:]
        force_data = self._get_ee_sum_ft()
        
        # we add this force data for terminating the episode for safety, and not allow it to re-try near this position
        # at next step, just back to the origin pose, and then restart a new episode.
        if max(np.fabs(force_data)) > 5000 :
            return True

        # if np.any(np.greater(np.fabs(joint_vel), self.joint_vel_limits_upper)):
        #     print('==========robot joint velocity exceed the joint velocity limit')
        #     return True

        self.lock.acquire()
        if self.contact_flag_released is False:
            return_flag = self.contact_flag
            print("===========detected the contact!")
            self.contact_flag_released = True
            self.lock.release()
            # pause the simulation?
            return return_flag
        else:
            self.lock.release()

        # TODO: deprecated this task error. check task error
        task_translation_error = norm(np.array(end_pose_dist[:3]))
        task_rotation_error = ros_utils.distance_of_quaternion(end_pose_dist[3:7])
        epsilon = self.tolerance

        # early terminate the trial process, because the wrong direction
        if task_translation_error <= 0.03 and task_rotation_error > 0.1:
            return True
        else:
            return False

        if task_translation_error <= epsilon and task_rotation_error <= 200 * epsilon:
            return True
        else:
            return False

    def _contain_point(self, bound_pts, end_pos):
        new_pts = np.zeros(bound_pts.shape)

        for i in range(1, 8):
            new_pts[i] = bound_pts[i] - bound_pts[0]

        square_bound_pts = np.zeros((3,2))
        square_bound_pts[0][0] = min(new_pts[:,0])
        square_bound_pts[0][1] = max(new_pts[:,0])
        square_bound_pts[1][0] = min(new_pts[:,1])
        square_bound_pts[1][1] = max(new_pts[:,1])
        square_bound_pts[2][0] = min(new_pts[:,2])
        square_bound_pts[2][1] = max(new_pts[:,2])

        end_new = end_pos - bound_pts[0]

        if square_bound_pts[0][0] <= end_new[0] <= square_bound_pts[0][1] and \
           square_bound_pts[1][0] <= end_new[1] <= square_bound_pts[1][1] and \
           square_bound_pts[2][0] <= end_new[2] <= square_bound_pts[2][1]:
            return True
        else:
            return False


    def _get_jacobian(self, joint_state):
        dim_pose = 6
        num_dofs = len(joint_state)

        q = joint_state
        jacobian = ros_utils.get_jacobians(q, self.robot, 'base_link', 'wrist_3_link')
        J = np.zeros((dim_pose, num_dofs))
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                J[i,j] = jacobian[i,j]

        return J
    

    def _reset_back(self):
        """Resets the state of the environment before the next training session and returns an initial observation/state.
        NOTE: The problem cannot be solved at present, because the gazebo cannot set model configuration correctly!

        We should reset all the gazebo init environment, if we do a non-contact task, we just need reset the robot joint angles.
        But here we using it with contact task, then we should reset the environment, not just the robot joint angles.
        NOTE: This function will be called when the class object is destoryed.

        TODO: now reset can't reset the grasp pose, the object can't be held by the hand.
        We can only reset the pose, not the force. We should make the object keep in the hand.
        **A easier way is that make the object fixed with the hand.**
        """
        
        # we should stop our controllers firstly or send the initial joint angles to robot 
        # using joint trajectory controller. Or the robot will back to its current position after unpause the simulation

        # reset arm position


        # reset world first time
        # self._reset_world_scene() # commented for test

        # add following to test
        # stop controllers always failed!
        # we must stop controllers before pause gazebo physics,
        # or the controller_manager will block waiting for gazebo controllers
        print("try to stop controllers")
        return_status = self.__switch_ctrl.call(start_controllers=[],
                        stop_controllers=["arm_gazebo_controller", "joint_state_controller"],
                            strictness=SwitchControllerRequest.BEST_EFFORT)
        print(return_status)
        print("stop controllers!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.__pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))


        # reset world firstly
        rospy.loginfo("reset world")
        self.__reset_world()


        joint_names = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        joint_positions = [-1.8220516691974549, 0.5562956844532536, 0.23514608713011054, 0.40291452827333263, 0.24649587287350094, 2.7958636338450624]

        time.sleep(1)
        return_status = self.__set_model.call(model_name="robot", 
                              urdf_param_name="robot_description",
                              joint_names=joint_names, 
                              joint_positions=joint_positions)
        rospy.loginfo("set model config %s", return_status)
        print("set robot model successfully!")

        time.sleep(5)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        timer = Timer(0.0, self.__start_ctrl)
        timer.start()

        # reset robot
        # self._reset_robot_arm_pose() # commented for test

        # self._reset_hand_pose() # no hand, so deprecated

        # test joint state controller
        # joint_data = None
        # while joint_data is None:
        #     # try:
        #     rospy.loginfo('=============first reset robot joint states...')
        #     joint_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
        #     print(joint_data.position)
        ## end test

        # reset the manipulation objects and robot arm
        # self._reset_object_pose(self.__current_object)

        time.sleep(0.2)

        # we should reset again
        # reset world second time
        # self._reset_world_scene()   # commented for test
        # reset robot
        # self._reset_robot_arm_pose() # commented for test
        # self._reset_hand_pose()

        # reset the manipulation objects and robot arm
        # self._reset_object_pose(self.__current_object)


        rospy.loginfo('fininshed reset environment')

        # read data to observation

        # test joint state controller

        # joint_data = None
        # while joint_data is None:
        #     # try:
        #     rospy.loginfo('=============second reset robot joint states...')
        #     joint_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
        #     print(joint_data.position)

        ## end test


        # update `state`
        # state = self.discretize_observation(data, 5)
        state,_ = self._get_obs()
        self.state = state

        # (needed by gym) return the initial observation or state
        return np.array(state)
