# python related packages
import time
import numpy as np
from threading import Timer, Lock
from gym import error, spaces, utils
from . import ros_general_utils as ros_utils  # custom user defined ros utils
from numpy.linalg import inv, norm
from gym_gazebo_ros.envs.tiago_robot.tiago_v0 import TiagoEnv

# ros related data structure
from geometry_msgs.msg import Twist, WrenchStamped, Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # Used for publishing UR joint angles.
from control_msgs.msg import *  # control with action interface
from sensor_msgs.msg import LaserScan, JointState


# ros related function packages
import rospy
import transforms3d as tf3d
from gazebo_msgs.msg import ModelState, ContactState, ContactsState
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

# numpy array output format
np.set_printoptions(precision=3, suppress=True)

"""
Reach task version 1:
Different from v0 version, reaching the ball without orientation constraits and
observation space inlucde ee positon and distance
"""


class TiagoReachV1(TiagoEnv):
    def __init__(self):
        super(TiagoReachV1, self).__init__()

        # resolve for action space and observation space
        self.hand_joint_names = ['hand_thumb_joint', 'hand_index_joint', 'hand_mrl_joint']
        self.ctrl_joint_names = []
        self.__joint_pos_lower = []
        self.__joint_pos_upper = []
        self.__joint_vel_upper = []
        self.__joint_vel_lower = []
        self.__joint_force_limits = []

        for i, joint in enumerate(self.robot.joints):
            # choose joint groups, e.g. arm, hand, torso and so on
            # TODO: control more groups in future work. this demo just consider arm group
            if joint.joint_type != 'fixed' and (joint.name.startswith('arm') is True):  # skip hand related joints
                self.ctrl_joint_names.append(joint.name)
                self.__joint_pos_lower.append(
                    joint.limit.lower) if joint.limit.lower is not None else self.__joint_pos_lower.append(-np.inf)
                self.__joint_pos_upper.append(
                    joint.limit.upper) if joint.limit.upper is not None else self.__joint_pos_upper.append(np.inf)
                self.__joint_vel_lower.append(-joint.limit.velocity)
                self.__joint_vel_upper.append(joint.limit.velocity)
                self.__joint_force_limits.append(joint.limit.effort)

        print("joints controlled in this demo: {}".format(self.ctrl_joint_names))

        # TODO: change the corresponding items according to your task
        # position, action is incremental postion command
        self.action_lower = np.array(self.__joint_vel_lower) * self.control_period * .5
        self.action_upper = np.array(self.__joint_vel_upper) * self.control_period * .5
        self.action_space = spaces.Box(self.action_lower, self.action_upper)

        # (state)observation space: task oriented configuration
        # for this pick and place task states include robot end effector pose, relative pose to target
        # default setting: velocity and force info. is disabled

        self.ee_pos_lower = [-np.inf] * 3
        self.ee_pos_upper = [np.inf] * 3
        self.ee_relative_pose_lower = [-np.inf] * 6  # end effector pos and distance to target
        self.ee_relative_pose_upper = [np.inf] * 6

        # force state

        self.low_full_state = self.ee_relative_pose_lower
        self.high_full_state = self.ee_relative_pose_upper
        # self.low_full_state = np.concatenate((self.low_state, self.force_sensor_lower)) # end effector pos, vec and force.
        # self.high_full_state = np.concatenate((self.high_state, self.force_sensor_upper))
        self.observation_space = spaces.Box(np.array(self.low_full_state), np.array(self.high_full_state))

        # Initialize a target pose for reaching task, will be done in reset func.

        # define the number of time step for every step know the time, then it can compute something
        # depend on the time
        self.contact_flag_released = True
        self.contact_flag = False
        self.tolerance = 1e-2  # reaching error threshold
        print("finish setup tiago reaching V1 task env.")

    def _step(self, action):
        """
        Interact with env with policy learning with RL agent
        action: depend on the action space setting
        Return:  state, reward, and done status in this function
        The action command comes from an agent, which is an algorithm used for making decision
        """
        # Clip by veloctiy
        act_clip, curr_goal = self._action_clip(action)  # should we used incremental command for a?

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        if (action == act_clip).all():
            print("new action: {}".format(np.array(action)))
        else:
            print("new clip action: {}".format(act_clip))

        try:

            # Control with action client
            self.arm_pos_control_client.wait_for_server()
            rospy.loginfo('connected to robot arm controller server')

            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.ctrl_joint_names
            g.trajectory.points = [JointTrajectoryPoint(positions=curr_goal, velocities=[0] * len(action),
                                                        time_from_start=rospy.Duration(self.control_period))]
            self.arm_pos_control_client.send_goal(g)
            rospy.loginfo('send position to robot arm')

            # bug? wait for result blocking!
            # self.arm_pos_control_client.wait_for_result()
            time.sleep(self.control_period)
            # pause physics for data processing
            # self.pause_physics.call()
            rospy.loginfo('Execute velocity control for one step')
            result = self.arm_pos_control_client.get_state()
            rospy.loginfo('task done with state: ' + self._get_states_string(result))
        except KeyboardInterrupt:
            self.arm_pos_control_client.cancel_goal()

        # get joint data.
        trans, abs_pos, joint_pos, joint_vel = self._get_obs()

        # TODO: add get the filtered force sensor data
        self.state = trans + abs_pos
        self.joint_pos = joint_pos

        done = self.is_task_done(self.state, joint_vel)

        distance = np.sqrt(np.sum(np.array(trans[:3])**2))
        reward = max(0.0, 2.0 - distance)
        print("=================episode: %d, step: %d, reward : %.3f, current dist: %.3f"
              % (self.current_epi, self.time_step_index, reward, distance))

        # (needed by gym) we should return the state(or observation from state(function of state)), reward, and done status.
        # If the task completed, such as distance to target is d > = 0.001,
        # then return done= True, else return done = False. done depend on the terminal conditions partly.
        # NOTE: `reward` always depend on done and action, or state, so it always calculated finally.
        self.time_step_index += 1
        return np.array(self.state), reward, done, {}

    def _action_clip(self, action):
        """
        clip action in case of action is beyond of the current reaching range or touching the joint limits
        :param action: original action value
        :return: clipped action value
        """
        # velocity clip
        action = np.array(action)
        act_clip = action
        if np.any(action >= self.action_upper) or np.any(action <= self.action_lower):
            print("velocity bound clip")
            act_clip = np.clip(action, self.action_lower, self.action_upper)

        # position clip
        curr_joint_pos = np.array(self.joint_pos)
        curr_goal = act_clip + curr_joint_pos
        if np.any(curr_goal >= self.__joint_pos_upper) or np.any(curr_goal <= self.__joint_pos_lower):
            print("joint bound clip")
            curr_goal = np.clip(curr_goal, self.__joint_pos_lower, self.__joint_pos_upper).tolist()
            # actually used clipped action, we should use this info. to update our policy
            act_clip = curr_goal - curr_joint_pos
        return act_clip, curr_goal

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
        idx = [i for i, x in enumerate(joint_data.name) if x in self.ctrl_joint_names]
        joint_pos = [joint_data.position[i] for i in idx]
        joint_vel = [joint_data.velocity[i] for i in idx]

        # get end-effector position and distance to target and end-effector velocity
        # end_pose_vel is end effector pose and velocity, ee_absolute_translation is absolute position
        distance, ee_abs_pos = self.get_ee_state()

        # get wrist force sensor data if titanium robot is used
        if self.robot_name == 'titanium':
            force_data_msg = None
            while force_data_msg is None:
                rospy.loginfo('try to receive force sensor data...')
                force_data_msg = rospy.wait_for_message('/wrist_ft', WrenchStamped, timeout=5)

            force_data = [force_data_msg.wrench.force.x, force_data_msg.wrench.force.y, force_data_msg.wrench.force.z,
                          force_data_msg.wrench.torque.x, force_data_msg.wrench.torque.y, force_data_msg.wrench.torque.z]
            print('==================your force data is: {}'.format(force_data))
        else:
            force_data = []

        return distance, ee_abs_pos, joint_pos, joint_vel

    def get_ee_state(self):
        """
        Compute distance between end effector and its absolute position
        :return:
        """
        rospy.wait_for_service('/gazebo/get_link_state')

        if self.robot_name == 'steel':
            # print('End effector is a gripper...')
            try:
                end_state = self.get_link_pose_srv.call('tiago_steel::arm_7_link', "base_footprint").link_state
            except (rospy.ServiceException) as exc:
                print("/gazebo/get_link_state service call failed:" + str(exc))
        else:
            # print('End effector is a 5 finger hand....')
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

        # transform form tool frame to grasp frame
        if self.robot_name == 'steel':
            r1 = tf3d.quaternions.quat2mat([-0.5, 0.5, 0.5, 0.5])
            trans1 = [0, 0, 0.046]
            arm_tool_affine = tf3d.affines.compose(trans1, r1, np.ones(3))
            arm_tool_pose = np.dot(end_pose_affine, arm_tool_affine)

            r2 = tf3d.quaternions.quat2mat([0, 0.707, 0, -0.707])
            trans2 = [0.01, 0, 0]
            gripper_link_affine = tf3d.affines.compose(trans2, r2, np.ones(3))
            gripper_pose = np.dot(arm_tool_pose, gripper_link_affine)

            r3 = tf3d.quaternions.quat2mat([0.5, -0.5, 0.5, 0.5])
            trans3 = [0, 0, -0.12]
            grasp_link_affine = tf3d.affines.compose(trans3, r3, np.ones(3))
            end_pose_affine = np.dot(gripper_pose, grasp_link_affine)
        else:
            r = tf3d.quaternions.quat2mat([1, 0, 0, 0])
            trans = [0.13, 0.02, 0]
            grasp_link_affine = tf3d.affines.compose(trans, r, np.ones(3))
            end_pose_affine = np.dot(end_pose_affine, grasp_link_affine)

        ee_abs_pos = end_pose_affine[:3, 3].reshape(-1).tolist()
        ee_quat = tf3d.quaternions.mat2quat(end_pose_affine[:3, :3]).tolist()

        # compute the relative pose to target pose (target frame relative the current frame)
        target= self.ee_target_pose[:3, 3].reshape(-1)
        distance= target - ee_abs_pos

        # form the end-effector twist list
        ee_velocity= [ee_vel_msg.linear.x, ee_vel_msg.linear.y, ee_vel_msg.linear.z,
                       ee_vel_msg.angular.x, ee_vel_msg.angular.y, ee_vel_msg.angular.z]

        return distance.tolist(), ee_abs_pos

    def _reset(self, random=False):
        # we should stop our controllers firstly or send the initial joint angles to robot
        # using joint trajectory controller. Or the robot will back to its current position after unpause the simulation

        # reset arm position
        print("=====================================================================================================\n")
        rospy.loginfo('reset environment...')

        # reset robot first stage
        self.reset_world()
        self.ee_target_pose, self.goal= self.spawn_dynamic_reaching_goal('ball', random)
        self._virtual_reset_arm_config()

        # self._reset_hand_pose() # no hand, so deprecated
        time.sleep(0.2)

        print('===========================================reset done===============================================\n')

        # read data to observation

        # update `state`
        # state = self.discretize_observation(data, 5)
        trans, abs_pos, joint_pos, joint_vel= self._get_obs()
        self.state= trans + abs_pos
        self.joint_pos= joint_pos
        self.current_epi += 1
        return np.array(self.state)

    def spawn_dynamic_reaching_goal(self, model_name, random=False):
        """
        spawn an object in Gazebo and return its pose to robot
        :return:
        """
        # stop simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as exc:
            print("/gazebo/unpause_physics service call failed:" + str(exc))

        x= 0.8
        y= -0.1
        z= 1
        modelState= ModelState()
        modelState.model_name= model_name
        modelState.pose.orientation.x= 0
        modelState.pose.orientation.y= 0
        modelState.pose.orientation.z= 0
        modelState.pose.orientation.w= 1
        modelState.reference_frame= 'world'
        if random:
            modelState.pose.position.x= x + np.random.sample() * 0.6 - 0.3
            modelState.pose.position.y= y
            modelState.pose.position.z= z + np.random.sample() * 0.6 - 0.3
        else:
            modelState.pose.position.x= x
            modelState.pose.position.y= y
            modelState.pose.position.z= z

        self.set_model_state(modelState)

        Rotation= tf3d.quaternions.quat2mat([modelState.pose.orientation.x, modelState.pose.orientation.y,
                                              modelState.pose.orientation.z, modelState.pose.orientation.w])
        Translation= [modelState.pose.position.x, modelState.pose.position.y, modelState.pose.position.z]
        target_pose= tf3d.affines.compose(Translation, Rotation, np.ones(3))

        hand2ball_trans= [-0.35, 0.01, 0]
        hand2ball_rot= tf3d.euler.euler2mat(0, 0, 0)
        hand2ball_affine= tf3d.affines.compose(hand2ball_trans, hand2ball_rot, np.ones(3))
        hand_pose_mat= np.dot(target_pose, np.linalg.inv(hand2ball_affine))
        hand_translation= hand_pose_mat[:3, 3].reshape(-1)
        hand_quat= tf3d.quaternions.mat2quat(hand_pose_mat[:3, :3]) # quaternion [w, x, y, z]
        hand_pos= Pose()
        hand_pos.position.x= hand_translation[0]
        hand_pos.position.y= hand_translation[1]
        hand_pos.position.z= hand_translation[2]
        # hand_pos.orientation.w = hand_quat[0]
        # hand_pos.orientation.x = hand_quat[1]
        # hand_pos.orientation.y = hand_quat[2]
        # hand_pos.orientation.z = hand_quat[3]
        hand_pos.orientation.w= 1
        hand_pos.orientation.x= 0
        hand_pos.orientation.y= 0
        hand_pos.orientation.z= 0
        # hand_pose = modelState.pose  # different from the target pose
        # hand_pose.position.x = x - 0.1
        return hand_pose_mat, hand_pos

    def is_task_done(self, state, joint_vel):

        # extract end pose distance from state
        end_pose_dist= state[:3]

        # TODO: add collision detection to cancel wrong/bad/negative trial!
        # TODO: add end-effector force sensor data to terminate the trial

        # if np.any(np.greater(np.fabs(joint_vel), self.__joint_vel_upper)):
        #     print('DONE, robot joint velocity exceed the joint velocity limit')
        #     return True

        # TODO: deprecated this task error. check task error
        task_translation_error= norm(np.array(end_pose_dist))
        if task_translation_error <= self.tolerance or task_translation_error >= 0.7:
            print("DONE, too far away from the target")
            return True
        else:
            return False