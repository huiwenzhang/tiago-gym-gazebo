# python related packages
import time
import numpy as np
from threading import Timer, Lock
from gym import error, spaces, utils
from gym.utils import seeding
from . import ros_general_utils as ros_utils # custom user defined ros utils
from numpy.linalg import inv, norm
import subprocess, os
from gym_gazebo_ros.envs import gazebo_env

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



class TiagoEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        """
        Initialized Tiago robot Env
        """
        super(TiagoEnv, self).__init__()

        # Controllers
        rospy.wait_for_service("/controller_manager/switch_controller")
        self.switch_ctrl = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)

        # param_names = rospy.get_param_names() # return a list include all the names in parameter serve
        arm_joint_names = rospy.get_param("/arm_controller/joints")
        print('Getting parameters from parameter server done...')


        # Tiago  controllers, actually here we use position controller
        self.arm_vel_publisher = rospy.Publisher("arm_controller/command", JointTrajectory, queue_size=1)
        self.torso_vel_publisher = rospy.Publisher("torso_controller/command", JointTrajectory, queue_size=1)
        self.head_vel_publisher = rospy.Publisher("head_controller/command", JointTrajectory, queue_size=1)
        self.base_vel_publisher = rospy.Publisher("mobile_base_controller/cmd_vel", Twist, queue_size=1)

        if self.robot_name == 'titanium':
            self.hand_vel_publisher = rospy.Publisher("hand_controller/command", JointTrajectory, queue_size=1)
        else:
            self.hand_vel_publisher = rospy.Publisher("gripper_controller/command", JointTrajectory, queue_size=1)


        # TIAGo action clients
        # TODO: we will replace the control api to a non-lib method using ros middle-ware
        self.arm_pos_control_client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
                                                                     FollowJointTrajectoryAction)
        self.torso_pos_control_client = actionlib.SimpleActionClient('/torso_controller/follow_joint_trajectory',
                                                                      FollowJointTrajectoryAction)
        self.head_pos_control_client = actionlib.SimpleActionClient('/head_controller/follow_joint_trajectory',
                                                                      FollowJointTrajectoryAction)
        self.head_point_control_client = actionlib.SimpleActionClient('/head_controller/point_head_action',
                                                                     PointHeadAction)
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)

        if self.robot_name == 'titanium':
            self.hand_pos_control_client = actionlib.SimpleActionClient('/hand_controller/follow_joint_trajectory',
                                                                         FollowJointTrajectoryAction)
        else:
            self.hand_pos_control_client = actionlib.SimpleActionClient('/gripper_controller/follow_joint_trajectory',
                                                                         FollowJointTrajectoryAction)


        # TODO: add objects into the world
        # This is an empty world, so no ojects need to be reset. We will include this in the future take and place task

        # (needed by gym) define action space, observation space, reward range
        # using URDF setup action_space and observation space, reward.
        self.robot = URDF.from_parameter_server()
        self.all_joint_names_order = [self.robot.joints[i].name for i in range(len(self.robot.joints))]

        # record non-fixed joints information
        # ==================only declare joints want to control in the demo, default:arm joints============
        self.hand_joint_names = ['hand_thumb_joint', 'hand_index_joint', 'hand_mrl_joint']
        self.ctrl_joint_names = []
        self.__joint_limits_lower = []
        self.__joint_limits_upper = []
        self.__joint_vel_limits_upper = []
        self.__joint_vel_limits_lower = []
        self.__joint_force_limits = []

        for i, joint in enumerate(self.robot.joints):
            # choose joint groups, e.g. arm, hand, torso and so on
            # TODO: control more groups in future work. this demo just consider arm group
            if joint.joint_type != 'fixed' and (joint.name.startswith('arm') is True): # skip hand related joints
                self.ctrl_joint_names.append(joint.name)
                self.__joint_limits_lower.append(joint.limit.lower) if joint.limit.lower is not None else self.__joint_limits_lower.append(-np.inf)
                self.__joint_limits_upper.append(joint.limit.upper) if joint.limit.upper is not None else self.__joint_limits_upper.append(np.inf)
                self.__joint_vel_limits_lower.append(-joint.limit.velocity)
                self.__joint_vel_limits_upper.append(joint.limit.velocity)
                self.__joint_force_limits.append(joint.limit.effort)


        print("joints controlled in this demo: {}".format(self.ctrl_joint_names))


        # TODO: change the corresponding items according to your task
        # position, action is postion command
        self.control_period = 0.01
        self.action_lower = np.array(self.__joint_limits_lower)
        self.action_upper = np.array(self.__joint_limits_upper)
        self.action_space = spaces.Box(self.action_lower, self.action_upper)

        # (state)observation space: task oriented configuration
        # for this pick and place task states include robot end effector pose, relative pose to target
        # default setting: velocity and force info. is disabled

        self.ee_pos_lower = [-np.inf] *3
        self.ee_pos_upper = [np.inf]*3
        self.ee_relative_pose_lower = [-np.inf]*3 + [-1]*4 # position [x,y,z] and quanternion (w,x,y,z)
        self.ee_relative_pose_upper = [np.inf]*3 + [1] * 4

        if self.robot_name == 'titanium':
            # include force info. if titanium is used
            self.force_sensor_lower = [-np.inf]*6
            self.force_sensor_upper = [np.inf]*6
        else:
            self.force_sensor_lower = []
            self.force_sensor_upper = []

        self.low_full_state = self.ee_relative_pose_lower
        self.high_full_state = self.ee_relative_pose_upper
        # self.low_full_state = np.concatenate((self.low_state, self.force_sensor_lower)) # end effector pos, vec and force.
        # self.high_full_state = np.concatenate((self.high_state, self.force_sensor_upper))
        self.observation_space = spaces.Box(np.array(self.low_full_state), np.array(self.high_full_state))

        # TODO: change this according to the task
        self.reward_range = (0, np.inf)

        self.state = None # observation space
        self.joint_pos = None # current joint position


        # define the number of time step for every step know the time, then it can compute something
        # depend on the time
        self.time_step_index = 0
        self.lock = Lock()
        self.tolerance = 1e-3 # reaching error threshold


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



    def _step(self, action):
        """
        Perform some action in the environment
        action: depend on the action space setting
        Return:  state, reward, and done status in this function
        The action command comes from an agent, which is an algorithm used for making decision
        """

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")


        try:
            # TODO: actually our action should consider the robot joint limit (including velocity limit)
            # TODO: add action prediction and corresponding terminate condition prediction before take action
            # TODO: limit action (position translocation), keep every step have a very small moving.
            # we use joint position increment to send to robot
            action = np.clip(action, self.action_lower, self.action_upper).tolist()
            # positions_point = np.clip(action + np.array(joint_data.position),self.__joint_limits_lower, self.__joint_limits_upper)
            print("new action is : {}".format(action))


            # Control with action client
            self.arm_pos_control_client.wait_for_server()
            rospy.loginfo('connected to robot arm controller server')

            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.ctrl_joint_names
            g.trajectory.points = [JointTrajectoryPoint(positions=action, velocities=[0]*len(action), time_from_start=rospy.Duration(self.control_period))]
            self.arm_pos_control_client.send_goal(g)
            rospy.loginfo('send position to robot arm')


            # bug? wait for result blocking!
            # self.arm_pos_control_client.wait_for_result()
            time.sleep(self.control_period)
            rospy.loginfo('Execute velocity control for one step')
            result = self.arm_pos_control_client.get_state()
            rospy.loginfo('task done with state: ' + self._get_states_string(result))

        except KeyboardInterrupt:
            self.arm_pos_control_client.cancel_goal()

        # get joint data.
        ee_rel_pose, force,  abs_pos, joint_position, joint_vel = self._get_obs()
        state = ee_rel_pose


        # TODO: add get the filtered force sensor data
        self.state = state
        self.joint_pos = joint_position
        # current_jacobian = self._get_jacobian(joint_position)

        # (needed by gym )done is the terminate condition. We should
        # return this value by considering the terminal condition.
        # TODO:  timestep stop set in max_episode_steps when register this env. 
        # Another is using --nb-rollout-steps in running this env example?
        done = self.is_task_done(state, joint_vel)



        # TODO: add jacobian cost as part of reward, like GPS, which can avoid the robot explore the sigularity position.
        #  Also can add joint torque as part of reward, like GPS. 
        # reward = max(0.0, 1.5 - 0.01*norm(np.array(end_pose_dist))**2)
        end_pose_dist = state[:7]
        distance = np.sqrt( np.sum(np.array(end_pose_dist[:3])**2) + ros_utils.distance_of_quaternion(end_pose_dist[3:7])**2 )
        # TODO: we should add safety bounds information to reward, not only terminate condition,like some paper!
        reward = max(0.0, 4.0 - distance)

        print("step: {}, reward : {}, current dist: {}".format(self.time_step_index, reward, distance))

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


    def _get_obs(self):
        """
        Return the current observation of the robot and env. e.g. end effector pos, target
        pose, environment objects state, images state
        :return:
        """
        joint_data = None
        while joint_data is None:
            rospy.loginfo('try to receive robot joint states...')
            # joint state topic return sensor_msgs/joint_state msg with member pos, vec and effort
            joint_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)

        # get joint position and velocity
        idx = [i for i, x in enumerate(joint_data.name) if x in self.ctrl_joint_names]
        joint_pos = [joint_data.position[i] for i in idx]
        joint_vel = [joint_data.velocity[i] for i in idx]

        # get end-effector position and distance to target and end-effector velocity
        # end_pose_vel is end effector pose and velocity, ee_absolute_translation is absolute position
        ee_relative_pose, ee_abs_pos = self.get_ee_state()

        # get wrist force sensor data if titanium robot is used
        if self.robot_name == 'titanium':
            force_data_msg = None
            while force_data_msg is None:
                rospy.loginfo('try to receive force sensor data...')
                force_data_msg = rospy.wait_for_message('/wrist_ft', WrenchStamped, timeout=5)

            ee_force = [force_data_msg.wrench.force.x, force_data_msg.wrench.force.y, force_data_msg.wrench.force.z,
                          force_data_msg.wrench.torque.x, force_data_msg.wrench.torque.y, force_data_msg.wrench.torque.z]
            print('==================your force data is: {}'.format(force_data))
        else:
            ee_force = []

        return ee_relative_pose, ee_force,  ee_abs_pos, joint_pos, joint_vel


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
        # TODO: reset env is needed
        # self.ee_target_pose, self.goal =  self.spawn_dynamic_reaching_goal('ball', random)
        self._virtual_reset_arm_config()
        # self._reset_hand_pose()

        # self._reset_hand_pose() # no hand, so deprecated

        # reset the manipulation objects and robot arm
        # self._reset_target_pose(self.__current_object)

        time.sleep(0.2)

        # we should reset again (second stage)
        # reset world second time
        # self._reset_world_scene()   # commented for test
        # reset robot
        # self._reset_robot_arm_pose() # commented for test
        # self._reset_hand_pose()
        
        # reset the manipulation objects and robot arm
        # self._reset_target_pose(self.__current_object)

        # # switch controller
        # self._switch_ctrl.call(stop_controllers=["arm_gazebo_controller"],
        #                 start_controllers=["joint_group_velocity_controller"],
        #                     strictness=SwitchControllerRequest.BEST_EFFORT)

        print('===========================================reset done===============================================\n')

        # read data to observation

        # update `state`
        # state = self.discretize_observation(data, 5)
        ee_rel_pos, ee_force,  abs_pos, joint_pos, joint_vel = self._get_obs()
        state = ee_rel_pos
        self.state = state
        self.joint_pos = joint_pos
        self.time_step_index = 0

        # (needed by gym) return the initial observation or state
        return np.array(state)


    def _reset_world_scene(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))

        # this will reset the simulation time, and this will affect the ros controllers
        rospy.loginfo("reset world")
        self.reset_world()

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))


    def _reset_arm_pose_with_play_motion(self):
        """
        Reset the pose of robot arm in gazebo to home pose.
        We will reset the robot arm pose use play motion package.
        """

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

       
        rospy.loginfo('wait for play motion action server')
        self.play_motion_client.wait_for_server()
        rospy.loginfo('connected to robot play motion server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal
        self.play_motion_client.cancel_goal()

        g = PlayMotionGoal()
        g.motion_name = 'unfold_arm'
        g.skip_planning = False
        rospy.loginfo('Sending goal with motion: ' + g.motion_name)
        self.play_motion_client.send_goal(g)
        rospy.loginfo('Waiting for results (reset)...')
        reset_ok = self.play_motion_client.wait_for_result(rospy.Duration(10))
        state = self.play_motion_client.get_state()

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

    

    def _reset_hand_pose(self):
        """Reset hand pose in gazebo.

        Reset the hand pose in gazebo. Using play motion pre-defined postures
        """
        print("===================reset hand pose================")
        rospy.loginfo('wait for play motion action server')
        self.play_motion_client.wait_for_server()
        rospy.loginfo('connected to robot play motion server')

        # we can cancel the trajectory goal, if the controller cannot go to the previous goal
        self.play_motion_client.cancel_goal()

        g = PlayMotionGoal()
        g.motion_name = 'open_hand'
        g.skip_planning = True
        rospy.loginfo('Sending goal with motion: ' + g.motion_name)
        self.play_motion_client.send_goal(g)
        rospy.loginfo('Waiting for results (reset)...')
        state = self.play_motion_client.get_state()

        rospy.loginfo('Reset hand motion done with state: ' + self._get_states_string(state))


    def _set_model_pose(self, model_name, model_pose, model_velocity=[0.0]*6, reference_frame='world'):
        """
        Set model pose attribute in Gazebo

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
        self.set_model_state(model_state_msg)


    def _start_ctrl(self):

        rospy.loginfo("STARTING CONTROLLERS")
        self.switch_ctrl.call(start_controllers=["arm_gazebo_controller", "joint_state_controller"],
                              stop_controllers=[], strictness=1)


    def _reset_target_pose(self, target_object_name):
        """
        Reset the pose of objects which are interacting with robot.
        :param target_object_name: objects want to reset for next episode training
        :return:
        """
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause_physics()
        # except (rospy.ServiceException) as exc:
        #     print("/gazebo/unpause_physics service call failed:" + str(exc))
        pass


    def get_ee_state(self):
        """Get end effector pose relative to the target.
        
        Return target pose relative to the target.
        """
        rospy.wait_for_service('/gazebo/get_link_state')

        if self.robot_name == 'steel':
            print('End effector is a gripper...')
            try:
                end_state = self.get_link_pose_srv.call('tiago_steel::gripper_grasp_link', "base_footprint").link_state
            except (rospy.ServiceException) as exc:
                print("/gazebo/get_link_state service call failed:" + str(exc))
        else:
            print('End effector is a 5 finger hand....')
            try:
                end_state = self.get_link_pose_srv.call('tiago_titanium::hand_mrl_link', "base_footprint").link_state
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


    def is_task_done(self, state, joint_vel):

        # extract end pose distance from state
        end_pose_dist =  state[:7]

        # TODO: add collision detection to cancel wrong/bad/negative trial!
        # TODO: add end-effector force sensor data to terminate the trial


        if np.any(np.greater(np.fabs(joint_vel), self.__joint_vel_limits_upper)):
            print('ERROR: velocity beyond system limitation, task is over')
            return True


        # TODO: deprecated this task error. check task error
        task_translation_error = norm(np.array(end_pose_dist[:3]))
        task_rotation_error = ros_utils.distance_of_quaternion(end_pose_dist[3:7])
        epsilon = self.tolerance


        if task_translation_error <= epsilon and task_rotation_error <= 200 * epsilon:
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
        return_status = self.switch_ctrl.call(start_controllers=[],
                                              stop_controllers=["arm_controller"],
                                              strictness=SwitchControllerRequest.BEST_EFFORT)
        print(return_status)
        print("stop arm controllers!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))

        # reset world firstly
        # rospy.loginfo("reset world")
        # self.__reset_world()

        joint_names = self.ctrl_joint_names
        joint_positions = [0.21, -0.2, -2.2, 1.15, -1.57, 0.2, 0.0]

        time.sleep(1)
        return_status = self.set_model.call(model_name='tiago_' + self.robot_name,
                                            urdf_param_name="robot_description",
                                            joint_names=joint_names,
                                            joint_positions=joint_positions)
        rospy.loginfo("reset arm state %s", return_status)

        time.sleep(5)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        self.switch_ctrl.call(start_controllers=["arm_controller"],
                              stop_controllers=[],
                              strictness=SwitchControllerRequest.BEST_EFFORT)

    def reach_to_point(self, goal=None):
        # TODO: change the target as needed
        """
        fulfill reaching task with traditional planing methods
         in Cartesian coordinates frame using moveit
        make sure the moveit_plan_service package is strarted before using this func.
        :return:
        """
        if goal == None:
            print("Use default pose as target...")
            goal = self.goal

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        rospy.wait_for_service("/cartesian_arm_plan", 5)
        self.cartesian_plan = rospy.ServiceProxy("/cartesian_arm_plan", CartesianArmPlan)
        print("Plan to goal position with MoveIt planner")

        header = Header()
        header.frame_id = "base_footprint"
        goal_pose = PoseStamped()
        goal_pose.pose = goal
        goal_pose.header = header

        self.cartesian_plan(goal_pose)





