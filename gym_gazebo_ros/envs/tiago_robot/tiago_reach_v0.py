# python related packages
import gym
import time
import numpy as np
from threading import Timer, Lock
from gym import error, spaces, utils
from gym.utils import seeding
from . import ros_general_utils as ros_utils # custom user defined ros utils
from numpy.linalg import inv, norm

# ros related data structure
from geometry_msgs.msg import Twist, WrenchStamped, Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing UR joint angles.
from control_msgs.msg import * # control with action interface
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty
import moveit_msgs.msg
# custom defined service call for moveit plan
from moveit_plan_service.srv import *


# ros related function packages
import rospy
import actionlib
# import moveit_commander
import transforms3d as tf3d
from gazebo_msgs.srv import GetModelState, GetLinkState, SetModelConfiguration, DeleteModel, SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState, ContactState, ContactsState
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from urdf_parser_py.urdf import URDF
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal



# from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

# TODO: we should create a command interface(based on ros middle-ware, topic, service or action)
# between learning module and robot manipulation module.
# But for simplify the program, we leave this work to do. Never do the robot things here! Keep a very pure thing
# about learning here!


class TiagoReachEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_node=True):
        """Initialized Tiago robot Env
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

        # self.__joint_state_sub = rospy.Subscriber("/joint_states", JointState,
        #                                           self.__joint_state_cb, queue_size=1)


        self.__robot_name = rospy.get_param("/robot_name")    # steel/titanium


        rospy.wait_for_service("/gazebo/reset_world", 10.0)
        self.__reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        rospy.wait_for_service("/gazebo/get_model_state", 10.0)
        self.__get_pose_srv = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState)

        # get the relative state
        rospy.wait_for_service("/gazebo/get_link_state", 10.0)
        self.__get_link_pose_srv = rospy.ServiceProxy(
            "/gazebo/get_link_state", GetLinkState)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.__pause_physics = rospy.ServiceProxy(
            "/gazebo/pause_physics", Empty)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.__unpause_physics = rospy.ServiceProxy(
            "/gazebo/unpause_physics", Empty)


        # ??
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

        print("ROS services setup done...")

        # Get parameters from parameter server
        # param_names = rospy.get_param_names() # return a list include all the names in parameter serve
        arm_joint_names = rospy.get_param("/arm_controller/joints")
        print('Getting parameters from parameter server done...')


        # use for tasks with contacts
        # rospy.Subscriber("/robot_fixed_box_bumper", ContactsState, self._robot_fixed_box_contact_callback)
        # self.lock = Lock()
        # self.contact_flag_released = True
        # self.contact_flag = False


        # Tiago velocity controllers
        self.arm_vel_publisher = rospy.Publisher("arm_controller/command", JointTrajectory, queue_size=1)
        self.torso_vel_publisher = rospy.Publisher("torso_controller/command", JointTrajectory, queue_size=1)
        self.head_vel_publisher = rospy.Publisher("head_controller/command", JointTrajectory, queue_size=1)
        self.base_vel_publisher = rospy.Publisher("mobile_base_controller/cmd_vel", Twist, queue_size=1)

        if self.__robot_name == 'titanium':
            self.hand_vel_publisher = rospy.Publisher("hand_controller/command", JointTrajectory, queue_size=1)
        else:
            self.hand_vel_publisher = rospy.Publisher("gripper_controller/command", JointTrajectory, queue_size=1)


        # TIAGo action clients
        # TODO: we will replace the control api to a non-lib method using ros middle-ware
        self.__arm_joint_traj_client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
                                                                     FollowJointTrajectoryAction)
        self.__torso_joint_traj_client = actionlib.SimpleActionClient('/torso_controller/follow_joint_trajectory',
                                                                      FollowJointTrajectoryAction)
        self.__head_joint_traj_client = actionlib.SimpleActionClient('/head_controller/follow_joint_trajectory',
                                                                      FollowJointTrajectoryAction)
        self.__head_point_client = actionlib.SimpleActionClient('/head_controller/point_head_action',
                                                                     PointHeadAction)

        self.__play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)

        if self.__robot_name == 'titanium':
            self.__hand_joint_traj_client = actionlib.SimpleActionClient('/hand_controller/follow_joint_trajectory',
                                                                         FollowJointTrajectoryAction)
        else:
            self.__hand_joint_traj_client = actionlib.SimpleActionClient('/gripper_controller/follow_joint_trajectory',
                                                                         FollowJointTrajectoryAction)


        # TODO: add objects into the world
        # This is an empty world, so no ojects need to be reset. We will include this in the future take and place task
        # self.__current_object = '50-cylinder-peg-gap-3mm'

        # (needed by gym) define action space, observation space, reward range
        # using URDF setup action_space and observation space, reward.
        self.robot = URDF.from_parameter_server()
        self.all_joint_names_order = [self.robot.joints[i].name for i in range(len(self.robot.joints))]

        # record non-fixed joints information
        # ==================only declare joints want to control in the demo, default:arm joints============
        self.__hand_joint_names = ['hand_thumb_joint', 'hand_index_joint', 'hand_mrl_joint']
        self.__arm_joint_names = []
        self.__joint_limits_lower = []
        self.__joint_limits_upper = []
        self.__joint_vel_limits_upper = []
        self.__joint_vel_limits_lower = []
        self.__joint_force_limits = []

        for i, joint in enumerate(self.robot.joints):
            # choose joint groups, e.g. arm, hand, torso and so on
            # TODO: control more groups in future work. this demo just consider arm group
            if joint.joint_type != 'fixed' and (joint.name.startswith('arm') is True): # skip hand related joints
                self.__arm_joint_names.append(joint.name)
                self.__joint_limits_lower.append(joint.limit.lower) if joint.limit.lower is not None else self.__joint_limits_lower.append(-np.inf)
                self.__joint_limits_upper.append(joint.limit.upper) if joint.limit.upper is not None else self.__joint_limits_upper.append(np.inf)
                self.__joint_vel_limits_lower.append(-joint.limit.velocity)
                self.__joint_vel_limits_upper.append(joint.limit.velocity)
                self.__joint_force_limits.append(joint.limit.effort)





        # here we know is:
        # ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        # we follow this order in joint velocity controller and joint state publisher
        print("joints controlled in this demo is: {}".format(self.__arm_joint_names))



        self.control_period = 0.01
        # self.action_space = spaces.Box(np.array(self.__joint_limits_lower), np.array(self.__joint_limits_upper))
        # 10.0 is used to make the action step more smaller
        self.action_lower = np.array(self.__joint_limits_lower)
        self.action_upper = np.array(self.__joint_limits_upper)
        self.action_space = spaces.Box(self.action_lower, self.action_upper)

        # observation space: include end effector pose, relative pose and force
        # velocity information is removed

        self.ee_relative_pose_lower = [-np.inf]*3 + [-1]*4 # position [x,y,z] and quanternion (w,x,y,z)
        self.ee_relative_pose_upper = [np.inf]*3 + [1] * 4
        # self.ee_pos_lower = [-np.inf] *3
        # self.ee_pos_upper = [np.inf]*3
        self.low_state = np.array(self.ee_relative_pose_lower)
        self.high_state = np.array(self.ee_relative_pose_upper)

        if self.__robot_name == 'titanium':
            self.force_sensor_lower = [-np.inf]*6
            self.force_sensor_upper = [np.inf]*6
        else:
            self.force_sensor_lower = []
            self.force_sensor_upper = []
        self.low_full_state = np.concatenate((self.low_state, self.force_sensor_lower)) # end effector pos, vec and force.
        self.high_full_state = np.concatenate((self.high_state, self.force_sensor_upper))
        # self.observation_space = spaces.Box(np.array(self.low_state), np.array(self.high_state))
        self.observation_space = spaces.Box(np.array(self.low_full_state), np.array(self.high_full_state))
        self.reward_range = (0, np.inf)

        self.state = None # observation space
        self.joint_pos = None # current joint position


        # Initialize a target pose for reaching task
        # TODO: replace this codes later with spawn_dynamic_target func
        # Rotation = tf3d.euler.euler2mat(0.473038,-1.151761, -0.364322, 'sxyz')
        # Translation = [0.466187,  0.428941, 1.418610]
        # self.ee_target_pose = tf3d.affines.compose(Translation, Rotation, np.ones(3))
        self.ee_target_pose, target_pose = self.spawn_dynamic_reaching_goal('ball')
        self.goal = target_pose # target pose, data type: geometry_msgs.msg.Pose

        # define the number of time step for every step know the time, then it can compute something
        # depend on the time
        self.time_step_index = 0
        self.lock = Lock()
        self.contact_flag_released = True
        self.contact_flag = False
        self.tolerance = 1e-3 # reaching error threshold

        # self.max_episode_steps = 10000 # only used for some algorithm, thi is complicated?


        # (needed by gym) seed and reset
        # self._reset_arm_pose_with_play_motion()
        self._seed()
        # self._reset()

        print("finish setup tiago env.")


    def _wait_for_valid_time(self, timeout):
        """
        wait for valid time(nonzero), this is important when using a simulated clock
        :param timeout:
        :return:
        """
        start_time = time.time()
        while not rospy.is_shutdown():
            if not rospy.Time.now().is_zero():
                return
            if time.time() - start_time > timeout:
                rospy.logerr("Time out waiting for valid time")
                exit(0)
            time.sleep(0.1)
        exit(0)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _robot_fixed_box_contact_callback(self, bumperstate):
        self.lock.acquire()
        if len(bumperstate.states) >= 1:
            self.contact_flag = True
            self.contact_flag_released = False
        self.lock.release()


    def _step(self, action):
        """
        Perform some action in the environment

        action: depend on the action space setting

        Return:  state, reward, and done status in this function

        The action command comes from an agent, which is an algorithm used for making decision
        """
        # if not self._valid_action_check(action):
        #     print("=======================Step: {}, current action is not valid, try next one".format(self.time_step_index))
        #     self.time_step_index += 1
        #     return np.array(self.state), 0, False, {}

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")


        self.lock.acquire() # acquire thread to main the current shared buffer
        contact_flag = self.contact_flag
        contact_flag_released = self.contact_flag_released
        self.lock.release()

        if contact_flag_released is False:
            if contact_flag is True:
                state, abs_pos,  joint_position, joint_vel = self._get_obs()
                # stop move
                rospy.loginfo("==== release contact state...")

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
                action = np.clip(action, self.action_lower, self.action_upper).tolist()
                # positions_point = np.clip(action + np.array(joint_data.position),self.__joint_limits_lower, self.__joint_limits_upper)
                print("============new action is : {}".format(action))


                # Control with topic
                # goal = JointTrajectory()
                # point = JointTrajectoryPoint()
                # point.positions = action
                # point.velocities = [0]*len(action)
                # point.time_from_start = rospy.Duration(self.control_period)
                # goal.points.append(point)
                # goal.joint_names = self.__joint_names_order
                # self.arm_vel_publisher.publish(goal)

                # Control with action client
                self.__arm_joint_traj_client.wait_for_server()
                rospy.loginfo('connected to robot arm controller server')

                g = FollowJointTrajectoryGoal()
                g.trajectory = JointTrajectory()
                g.trajectory.joint_names = self.__arm_joint_names
                g.trajectory.points = [JointTrajectoryPoint(positions=action, velocities=[0]*len(action), time_from_start=rospy.Duration(self.control_period))]
                self.__arm_joint_traj_client.send_goal(g)
                rospy.loginfo('send position to robot arm')


                # bug? wait for result blocking!
                # self.__arm_joint_traj_client.wait_for_result()
                time.sleep(self.control_period)
                rospy.loginfo('Execute velocity control for one step')
                result = self.__arm_joint_traj_client.get_state()
                rospy.loginfo('task done with state: ' + self._get_states_string(result))

            except KeyboardInterrupt:
                self.__arm_joint_traj_client.cancel_goal()

        # get joint data.
        state, abs_pos, joint_position, joint_vel = self._get_obs()


        # TODO: add get the filtered force sensor data
        self.state = state
        self.joint_pos = joint_position
        # current_jacobian = self._get_jacobian(joint_position)

        # (needed by gym )done is the terminate condition. We should
        # return this value by considering the terminal condition.
        # TODO:  timestep stop set in max_episode_steps when register this env. 
        # Another is using --nb-rollout-steps in running this env example?
        done = self.is_task_done(state, joint_vel)



        rospy.loginfo('maximum pisode step is {}, current time step index is {}'.format(self.spec.timestep_limit, self.time_step_index))

        # TODO: add jacobian cost as part of reward, like GPS, which can avoid the robot explore the sigularity position.
        #  Also can add joint torque as part of reward, like GPS. 
        # reward = max(0.0, 1.5 - 0.01*norm(np.array(end_pose_dist))**2)
        end_pose_dist = state[:7]
        distance = np.sqrt( np.sum(np.array(end_pose_dist[:3])**2) + ros_utils.distance_of_quaternion(end_pose_dist[3:7])**2 )
        # TODO: we should add safety bounds information to reward, not only terminate condition,like some paper!
        reward = max(0.0, 2.0 - distance)

        print("===================reward is: {}==============================".format(reward))
        print("===================distance is : {}===========================".format(distance))

        # (needed by gym) we should return the state(or observation from state(function of state)), reward, and done status.
        # If the task completed, such as distance to target is d > = 0.001,
        # then return done= True, else return done = False. done depend on the terminal conditions partly.
        # NOTE: `reward` always depend on done and action, or state, so it always calculated finally.
        self.time_step_index += 1
        return np.array(self.state), reward, done, {}

    def _valid_action_check(self, action):
        """
        checking the validation of the position action command. Return true if the command can be executed by the robot with
        the maximum velocity constraints
        :param action:
        :return:
        """
        action = np.array(action)
        pos_upper = np.array(self.joint_pos) + np.array(self.__joint_vel_limits_upper) * self.control_period
        pos_lower = np.array(self.joint_pos) + np.array(self.__joint_vel_limits_lower) * self.control_period
        return  True if np.all(action <= pos_upper) and np.all(action >= pos_lower) else False


    def zero_velocity_stop(self):
        # used for velocity control
        # TODO:implement velocity controller later (use position controller and velocity controller)
        vel_msg = Float64MultiArray()

        vel_msg.data = [0.0] *6

        self.arm_vel_publisher.publish(vel_msg)


    def _get_obs(self):
        """
        Return the current observation of the robot and env. e.g. end effector pos, target object
        pose, environment objects state, images state
        :return:
        """
        joint_data = None
        while joint_data is None:
            rospy.loginfo('try to receive robot joint states...')
            # joint state topic return sensor_msgs/joint_state msg with member pos, vec and effort
            joint_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)

        # get joint position and velocity
        idx = [i for i, x in enumerate(joint_data.name) if x in self.__arm_joint_names]
        joint_position = [joint_data.position[i] for i in idx]
        joint_vel = [joint_data.velocity[i] for i in idx]

        # get end-effector position and distance to target and end-effector velocity
        # end_pose_vel is end effector pose and velocity, ee_absolute_translation is absolute position
        ee_relative_pose, ee_abs_pos = self.get_ee_state()

        # get wrist force sensor data if titanium robot is used
        if self.__robot_name == 'titanium':
            force_data_msg = None
            while force_data_msg is None:
                rospy.loginfo('try to receive force sensor data...')
                force_data_msg = rospy.wait_for_message('/wrist_ft', WrenchStamped, timeout=5)

            force_data = [force_data_msg.wrench.force.x, force_data_msg.wrench.force.y, force_data_msg.wrench.force.z,
                          force_data_msg.wrench.torque.x, force_data_msg.wrench.torque.y, force_data_msg.wrench.torque.z]
            print('==================your force data is: {}'.format(force_data))
        else:
            force_data = []

        state =  ee_relative_pose + force_data

        return state, ee_abs_pos, joint_position, joint_vel


    def _reset(self, random=False):
        # we should stop our controllers firstly or send the initial joint angles to robot
        # using joint trajectory controller. Or the robot will back to its current position after unpause the simulation

        # reset arm position
        print("=====================================================================================================\n")
        rospy.loginfo('reset environment...')
        # reset world first time(first stage)
        # self._reset_world_scene() # commented for test

        # add following to test

        # reset robot first stage
        # self.spawn_dynamic_reaching_goal('ball')
        # self._reset_arm_pose_with_play_motion()
        self.ee_target_pose, self.goal =  self.spawn_dynamic_reaching_goal('ball', random)
        self._virtual_reset_arm_config()
        # self._reset_hand_pose()

        # self._reset_hand_pose() # no hand, so deprecated

        # reset the manipulation objects and robot arm
        # self._reset_object_pose(self.__current_object)

        time.sleep(0.2)

        # we should reset again (second stage)
        # reset world second time
        # self._reset_world_scene()   # commented for test
        # reset robot
        # self._reset_robot_arm_pose() # commented for test
        # self._reset_hand_pose()
        
        # reset the manipulation objects and robot arm
        # self._reset_object_pose(self.__current_object)

        # # switch controller
        # self.__switch_ctrl.call(stop_controllers=["arm_gazebo_controller"],
        #                 start_controllers=["joint_group_velocity_controller"],
        #                     strictness=SwitchControllerRequest.BEST_EFFORT)
        print('===========================================reset done===============================================\n')

        # read data to observation

        # update `state`
        # state = self.discretize_observation(data, 5)
        state, abs_pos, joint_pos, joint_vel = self._get_obs()
        self.state = state
        self.joint_pos = joint_pos
        self.time_step_index = 0

        # (needed by gym) return the initial observation or state
        return np.array(state)

    def _render(self, mode='human', close=False):
        """Send state to plot in the window. This is not needed by gazebo env.

        Args:
            mode (str): render mode. 'human' for no render on the window
            close (bool): True for close the window, False for keep open

        Returns:
            return_type: return nothing.

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


    def _reset_arm_pose_with_play_motion(self):
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

        # self.__switch_ctrl.call(start_controllers=["arm_gazebo_controller"],
        #                 stop_controllers=["joint_group_velocity_controller"],
        #                     strictness=SwitchControllerRequest.BEST_EFFORT)

        # reset robot arm to lift pose using joint trajecotry controller
        rospy.loginfo('wait for play motion action server')
        self.__play_motion_client.wait_for_server()
        rospy.loginfo('connected to robot play motion server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal
        self.__play_motion_client.cancel_goal()

        g = PlayMotionGoal()
        g.motion_name = 'unfold_arm'
        g.skip_planning = False
        rospy.loginfo('Sending goal with motion: ' + g.motion_name)
        self.__play_motion_client.send_goal(g)
        rospy.loginfo('Waiting for results (reset)...')
        reset_ok = self.__play_motion_client.wait_for_result(rospy.Duration(10))
        state = self.__play_motion_client.get_state()

        if reset_ok:
            rospy.loginfo('Reset successfully with state: ' + self._get_states_string(state))
        else:
            rospy.loginfo('Reset failed with state: ' + self._get_states_string(state))


    def _get_states_string(self, status_code):
        """
        Convert a action executon result to a string
        :param status_code:
        :return:
        """
        return actionlib.GoalStatus.to_string(status_code)

    def _reset_arm_pose_with_controller(self):
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

        # reset robot arm to lift pose using joint trajecotry controller
        rospy.loginfo('wait for robot arm joint trajectory controller server')
        self.__arm_joint_traj_client.wait_for_server()
        rospy.loginfo('connected to robot arm controller server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal
        self.__arm_joint_traj_client.cancel_goal()

        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()

        g.trajectory.joint_names = self.__arm_joint_names
        slow_duration =1.0
        try:
            # Q_target = [1.057810,-0.683443,-0.532145,-1.944877,-1.570304,-2.100711]
            # Q_target = [1.38,-0.75,-0.38,-2.32,-1.7,-0.06]
            Q_target = self.home_arm_joints_position
            Q_target = np.clip(Q_target + self.np_random.uniform(-0.2,0.2,(6,)), self.__joint_limits_lower, self.__joint_limits_upper).tolist()
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

        Reset the hand pose in gazebo. Using play motion pre-defined postures
        """
        print("===================reset hand pose================")
        rospy.loginfo('wait for play motion action server')
        self.__play_motion_client.wait_for_server()
        rospy.loginfo('connected to robot play motion server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal
        self.__play_motion_client.cancel_goal()

        g = PlayMotionGoal()
        g.motion_name = 'open_hand'
        g.skip_planning = True
        rospy.loginfo('Sending goal with motion: ' + g.motion_name)
        self.__play_motion_client.send_goal(g)
        rospy.loginfo('Waiting for results (reset)...')
        state = self.__play_motion_client.get_state()

        rospy.loginfo('Reset hand motion done with state: ' + self._get_states_string(state))


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


    def __start_ctrl(self):

        rospy.loginfo("STARTING CONTROLLERS")
        self.__switch_ctrl.call(start_controllers=["arm_gazebo_controller", "joint_state_controller"],
                        stop_controllers=[], strictness=1)


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
        # model_name = self.__current_object
        # if self.__current_object == '50-cylinder-peg-gap-3mm':
        #     rospy.loginfo('reset cylinder peg')
        #     # set to default grasp position is not easy, because the robot re-grasp can't only depend on the robot hand.
        #     model_pose = [0.735317, -0.306257, 1.204858, -0.000388, -0.000580, -0.001478]
        #     self._set_object_model_pose(model_name=model_name, model_pose=model_pose)
        # elif self.__current_object == 'triangle-peg-with-3mm-gap':
        #     model_pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #     self._set_object_model_pose(model_name=model_name, model_pose=model_pose)
        # elif self.__current_object == '50-rectangle-3mm':
        #     model_pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #     self._set_object_model_pose(model_name=model_name, model_pose=model_pose)
        # elif self.__current_object == 'baseplate':
        #     rospy.loginfo('reset baseplate peg')
        #     model_pose = [0.735555, -0.306439, 1.093998, 0, 0, -1.573690]
        #     self._set_object_model_pose(model_name=model_name, model_pose=model_pose)

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

        if self.__robot_name == 'steel':
            print('End effector is a gripper...')
            try:
                end_state = self.__get_link_pose_srv.call('tiago_steel::gripper_grasp_link', "base_footprint").link_state
            except (rospy.ServiceException) as exc:
                print("/gazebo/get_link_state service call failed:" + str(exc))
        else:
            print('End effector is a 5 finger hand....')
            try:
                end_state = self.__get_link_pose_srv.call('tiago_titanium::hand_mrl_link', "base_footprint").link_state
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

        ee_abs_pos = end_pose_affine[:3, 3].reshape(-1).tolist()

        # compute the relative pose to target pose (target frame relative the current frame)
        ee_pose_dist_affine = np.dot(inv(end_pose_affine), self.ee_target_pose)

        # translate the relative pose affine matrix to list (position, quaternion)
        ee_translation = ee_pose_dist_affine[:3,3].reshape(-1).tolist()
        ee_quat = tf3d.quaternions.mat2quat(ee_pose_dist_affine[:3,:3]).tolist()
        ee_relative_pose = ee_translation + ee_quat

        # form the end-effector twist list
        ee_velocity = [ee_vel_msg.linear.x, ee_vel_msg.linear.y, ee_vel_msg.linear.z, 
                        ee_vel_msg.angular.x, ee_vel_msg.angular.y, ee_vel_msg.angular.z]

        return  ee_relative_pose, ee_abs_pos

    def spawn_dynamic_reaching_goal(self, model_name, random=False):
        """
        spawn an object in Gazebo and return its pose to robot
        :return:
        """
        # stop simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        x = 0.5
        y = -0.1
        z = 1
        modelState = ModelState()
        modelState.model_name = model_name
        modelState.pose.orientation.x = 0
        modelState.pose.orientation.y = 0
        modelState.pose.orientation.z = 0
        modelState.pose.orientation.w = 1
        modelState.reference_frame = 'world'
        if random:
            modelState.pose.position.x = x + np.random.sample()*0.6 - 0.3
            modelState.pose.position.y = y
            modelState.pose.position.z = z + np.random.sample()*0.6 - 0.3
        else:
            modelState.pose.position.x = x
            modelState.pose.position.y = y
            modelState.pose.position.z = z

        self.__set_model_state(modelState)

        Rotation = tf3d.quaternions.quat2mat([modelState.pose.orientation.x, modelState.pose.orientation.y,
                                              modelState.pose.orientation.z, modelState.pose.orientation.w])
        Translation = [modelState.pose.position.x, modelState.pose.position.y, modelState.pose.position.z]
        target_pose = tf3d.affines.compose(Translation, Rotation, np.ones(3))
        return target_pose, modelState.pose


    def is_task_done(self, state, joint_vel):

        # extract end pose distance from state
        end_pose_dist =  state[:7]

        # TODO: add collision detection to cancel wrong/bad/negative trial!
        # TODO: add end-effector force sensor data to terminate the trial

        # detect safety bound
        # is_in_bound = self._contain_point(self.safety_bound_points, end_absolute_translation)
        # if is_in_bound is False:
        #     print("=================robot out of box=====================")
        #     print("=================current end absolute pose is:=========================")
        #     print(end_absolute_translation)
        #     return True

        # extract end-effector force sensor data from state
        force_data = state[7:]

        # if force_data is not None and np.fabs(max(force_data)) > 5 :
        #     return True

        if np.any(np.greater(np.fabs(joint_vel), self.__joint_vel_limits_upper)):
            print('==========robot joint velocity exceed the joint velocity limit')
            return True

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

    def _virtual_reset_arm_config(self):
        """
        instead of reset arm pose with controller executed by action interface. Here we use virtual reset
        where arm joints configures are reset instantly
        """

        # we should stop our controllers firstly or send the initial joint angles to robot
        # using joint trajectory controller. Or the robot will back to its current position after unpause the simulation

        # reset arm position



        # add following to test
        # stop controllers always failed!
        # we must stop controllers before pause gazebo physics,
        # or the controller_manager will block waiting for gazebo controllers
        print("reset arm to home state")
        return_status = self.__switch_ctrl.call(start_controllers=[],
                                                stop_controllers=["arm_controller"],
                                                strictness=SwitchControllerRequest.BEST_EFFORT)
        print(return_status)
        print("stop arm controllers!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.__pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))

        # reset world firstly
        # rospy.loginfo("reset world")
        # self.__reset_world()

        joint_names = self.__arm_joint_names
        joint_positions = [0.21, -0.2, -2.2, 1.15, -1.57, 0.2, 0.0]

        time.sleep(1)
        return_status = self.__set_model.call(model_name='tiago_'+self.__robot_name,
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

        self.__switch_ctrl.call(start_controllers=["arm_controller"],
                                stop_controllers=[],
                                strictness=SwitchControllerRequest.BEST_EFFORT)

    def reach_to_point(self):
        """
        fulfill reaching task with traditional planing methods
        planing in Cartesian coordinates frame using moveit
        make sure the moveit_plan_service package is strarted before using this func.
        :return:
        """
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.__unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        rospy.wait_for_service("/cartesian_arm_plan", 5)
        self.cartesian_plan = rospy.ServiceProxy("/cartesian_arm_plan", CartesianArmPlan)
        print("Plan to goal position with MoveIt planner")

        header = Header()
        header.frame_id = "base_footprint"
        goal_pose = PoseStamped()
        goal_pose.pose = self.goal
        goal_pose.header = header

        self.cartesian_plan(goal_pose)



