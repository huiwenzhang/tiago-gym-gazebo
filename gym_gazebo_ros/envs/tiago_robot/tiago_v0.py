# python related packages
import time
import numpy as np
from threading import Timer, Lock
from gym.utils import seeding
from . import ros_general_utils as ros_utils  # custom user defined ros utils
from numpy.linalg import inv, norm
from gym_gazebo_ros.envs import gazebo_env

# ros related data structure
from geometry_msgs.msg import Twist, WrenchStamped, Pose, PoseStamped
# Used for publishing UR joint angles.
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import *  # control with action interface
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Float64MultiArray, Header
# custom defined service call for moveit plan
from moveit_plan_service.srv import *


# ros related function packages
import rospy
import actionlib
import transforms3d as tf3d
from gazebo_msgs.msg import ModelState, ContactState, ContactsState
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from urdf_parser_py.urdf import URDF
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

"""
A base class, define shared variables and methods for specific task
"""


class TiagoEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        super(TiagoEnv, self).__init__()

        # Controllers
        rospy.wait_for_service("/controller_manager/switch_controller")
        self.switch_ctrl = rospy.ServiceProxy(
            "/controller_manager/switch_controller", SwitchController)

        # param_names = rospy.get_param_names() # return a list include all the
        # names in parameter serve
        arm_joint_names = rospy.get_param("/arm_controller/joints")
        print('Getting parameters from parameter server done...')

        # Tiago  controllers, actually here we use position controller
        self.arm_vel_publisher = rospy.Publisher(
            "arm_controller/command", JointTrajectory, queue_size=1)
        self.torso_vel_publisher = rospy.Publisher(
            "torso_controller/command", JointTrajectory, queue_size=1)
        self.head_vel_publisher = rospy.Publisher(
            "head_controller/command", JointTrajectory, queue_size=1)
        self.base_vel_publisher = rospy.Publisher(
            "mobile_base_controller/cmd_vel", Twist, queue_size=1)

        if self.robot_name == 'titanium':
            self.hand_vel_publisher = rospy.Publisher(
                "hand_controller/command", JointTrajectory, queue_size=1)
        else:
            self.hand_vel_publisher = rospy.Publisher(
                "gripper_controller/command", JointTrajectory, queue_size=1)

        # TIAGo action clients
        # TODO: we will replace the control api to a non-lib method using ros
        # middle-ware
        self.arm_pos_control_client = actionlib.SimpleActionClient(
            '/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.torso_pos_control_client = actionlib.SimpleActionClient(
            '/torso_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.head_pos_control_client = actionlib.SimpleActionClient(
            '/head_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.head_point_control_client = actionlib.SimpleActionClient(
            '/head_controller/point_head_action', PointHeadAction)
        self.play_motion_client = actionlib.SimpleActionClient(
            '/play_motion', PlayMotionAction)

        # different on end effector setting for gripper and palm
        if self.robot_name == 'titanium':
            self.hand_pos_control_client = actionlib.SimpleActionClient(
                '/hand_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
            self.ee_link = 'hand_palm_link'
            self.ee_frame = 'hand_grasping_frame'
        else:
            self.hand_pos_control_client = actionlib.SimpleActionClient(
                '/gripper_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
            self.ee_link = 'gripper_link'
            self.ee_frame = 'gripper_grasping_frame'

        # contact information
        self.lock = Lock()
        rospy.Subscriber(
            "/gripper_left_finger_contact_state",
            ContactsState,
            self._contact_cb)
        rospy.Subscriber(
            "/gripper_right_finger_contact_state",
            ContactsState,
            self._contact_cb)

        self.robot = URDF.from_parameter_server()
        self.all_joints = [
            self.robot.joints[i].name for i in range(len(self.robot.joints))]
        self.all_movable_joints = []
        for i, joint in enumerate(self.robot.joints):
            if joint.joint_type != 'fixed':
                self.all_movable_joints.append(joint.name)

        #######################################################################
        ########### set task specific env in child class, initialize action and
        #######################################################################

        self.state = None  # observation space
        self.joint_pos = None  # current joint position

        # define the number of time step for every step know the time, then it can compute something
        # depend on the time
        self.time_step_index = 0
        self.current_epi = 0
        self.tolerance = 1e-2  # reaching error threshold
        self.control_period = 0.025
        self.contact_flag_released = True
        self.contact_flag = False

        print("finish setup parent class tiago env.")

    def _contact_cb(self, bumperstate):
        # multiple contacts
        self.lock.acquire()
        if len(bumperstate.states) >= 1:
            self.contact_flag = True
            self.contact_flag_released = False
            for i in range(len(bumperstate.states)):
                print(
                    bumperstate.states[i].collision1_name +
                    " collide with " +
                    bumperstate.states[i].collision2_name)
        self.lock.release()

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
        raise NotImplementedError

    def _valid_action_check(self, action):
        """
        checking the validation of the position action command. Return true if the command can be executed by the robot with
        the maximum velocity constraints
        :param action:
        :return:
        """
        action = np.array(action)
        pos_upper = np.array(
            self.joint_pos) + np.array(self.__joint_vel_limits_upper) * self.control_period
        pos_lower = np.array(
            self.joint_pos) + np.array(self.__joint_vel_limits_lower) * self.control_period
        return True if np.all(
            action <= pos_upper) and np.all(
            action >= pos_lower) else False

    def _get_obs(self):
        """
        Return the current observation of the robot and env. e.g. end effector pos, target
        pose, environment objects state, images state
        :return:
        """
        raise NotImplementedError

    def _reset(self, random=False):
        # we should stop our controllers firstly or send the initial joint angles to robot
        # using joint trajectory controller. Or the robot will back to its
        # current position after unpause the simulation

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

        time.sleep(0.2)

        print('===========================================reset done===============================================\n')

        # read data to observation

        # update `state`
        # state = self.discretize_observation(data, 5)
        ee_rel_pos, ee_force, abs_pos, joint_pos, joint_vel = self._get_obs()
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

        # this will reset the simulation time, and this will affect the ros
        # controllers
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

        # we can cancel the trajectory goal, if the controller cannot go to the
        # previous goal
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
            rospy.loginfo(
                'Reset successfully with state: ' +
                self._get_states_string(state))
        else:
            rospy.loginfo(
                'Reset failed with state: ' +
                self._get_states_string(state))

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

        # we can cancel the trajectory goal, if the controller cannot go to the
        # previous goal
        self.play_motion_client.cancel_goal()

        g = PlayMotionGoal()
        g.motion_name = 'open_hand'
        g.skip_planning = True
        rospy.loginfo('Sending goal with motion: ' + g.motion_name)
        self.play_motion_client.send_goal(g)
        rospy.loginfo('Waiting for results (reset)...')
        state = self.play_motion_client.get_state()

        rospy.loginfo(
            'Reset hand motion done with state: ' +
            self._get_states_string(state))

    def _set_model_pose(
            self,
            model_name,
            model_pose,
            model_velocity=[0.0] * 6,
            reference_frame='world'):
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
        self.switch_ctrl.call(
            start_controllers=[
                "arm_gazebo_controller",
                "joint_state_controller"],
            stop_controllers=[],
            strictness=1)

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
        raise NotImplementedError

    def is_task_done(self, state, joint_vel):
        raise NotImplementedError

    def _get_jacobian(self, joint_state):
        dim_pose = 6
        num_dofs = len(joint_state)

        q = joint_state
        jacobian = ros_utils.get_jacobians(
            q, self.robot, 'base_link', 'wrist_3_link')
        J = np.zeros((dim_pose, num_dofs))
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                J[i, j] = jacobian[i, j]

        return J

    def _virtual_reset_arm_config(self):
        """
        instead of reset arm pose with controller executed by action interface. Here we use virtual reset
        where arm joints configures are reset instantly
        """

        # we should stop our controllers firstly or send the initial joint angles to robot
        # using joint trajectory controller. Or the robot will back to its
        # current position after unpause the simulation

        # reset arm position

        # add following to test
        # stop controllers always failed!
        # we must stop controllers before pause gazebo physics,
        # or the controller_manager will block waiting for gazebo controllers
        print("reset arm to home state")
        return_status = self.switch_ctrl.call(
            start_controllers=[],
            stop_controllers=["arm_controller"],
            strictness=SwitchControllerRequest.BEST_EFFORT)
        print(return_status)
        print("stop arm controllers!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/pause_physics service call failed:" + str(exc))

        joint_names = self.ctrl_joint_names
        joint_positions = [0.21, -0.08, -1.24, 1.4, -1.8, 0.0, 0.0]

        time.sleep(1)
        return_status = self.set_model.call(
            model_name='tiago_' + self.robot_name,
            urdf_param_name="robot_description",
            joint_names=joint_names,
            joint_positions=joint_positions)
        rospy.loginfo("reset arm state %s", return_status)

        time.sleep(3)

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
        if goal is None:
            print("Use default pose as target...")
            goal = self.goal

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        rospy.wait_for_service("/cartesian_arm_plan", 5)
        self.cartesian_plan = rospy.ServiceProxy(
            "/cartesian_arm_plan", CartesianArmPlan)
        print("Plan to goal position with MoveIt planner")

        header = Header()
        header.frame_id = "base_footprint"
        goal_pose = PoseStamped()
        goal_pose.pose = goal
        goal_pose.header = header

        self.cartesian_plan(goal_pose)
