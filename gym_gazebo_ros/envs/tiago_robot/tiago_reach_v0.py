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


class TiagoReachEnv(TiagoEnv):
    def __init__(self):
        super(TiagoReachEnv, self).__init__()

        # resolve for action space and observation space
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
            if joint.joint_type != 'fixed' and (joint.name.startswith('arm') is True):  # skip hand related joints
                self.ctrl_joint_names.append(joint.name)
                self.__joint_limits_lower.append(
                    joint.limit.lower) if joint.limit.lower is not None else self.__joint_limits_lower.append(-np.inf)
                self.__joint_limits_upper.append(
                    joint.limit.upper) if joint.limit.upper is not None else self.__joint_limits_upper.append(np.inf)
                self.__joint_vel_limits_lower.append(-joint.limit.velocity)
                self.__joint_vel_limits_upper.append(joint.limit.velocity)
                self.__joint_force_limits.append(joint.limit.effort)

        print("joints controlled in this demo: {}".format(self.ctrl_joint_names))

        # TODO: change the corresponding items according to your task
        # position, action is postion command
        self.action_lower = np.array(self.__joint_limits_lower)
        self.action_upper = np.array(self.__joint_limits_upper)
        self.action_space = spaces.Box(self.action_lower, self.action_upper)

        # (state)observation space: task oriented configuration
        # for this pick and place task states include robot end effector pose, relative pose to target
        # default setting: velocity and force info. is disabled

        self.ee_pos_lower = [-np.inf] * 3
        self.ee_pos_upper = [np.inf] * 3
        self.ee_relative_pose_lower = [-np.inf] * 3 + [-1] * 4  # position [x,y,z] and quanternion (w,x,y,z)
        self.ee_relative_pose_upper = [np.inf] * 3 + [1] * 4

        if self.robot_name == 'titanium':
            # include force info. if titanium is used
            self.force_sensor_lower = [-np.inf] * 6
            self.force_sensor_upper = [np.inf] * 6
        else:
            self.force_sensor_lower = []
            self.force_sensor_upper = []

        self.low_full_state = self.ee_relative_pose_lower
        self.high_full_state = self.ee_relative_pose_upper
        # self.low_full_state = np.concatenate((self.low_state, self.force_sensor_lower)) # end effector pos, vec and force.
        # self.high_full_state = np.concatenate((self.high_state, self.force_sensor_upper))
        self.observation_space = spaces.Box(np.array(self.low_full_state), np.array(self.high_full_state))

        # Initialize a target pose for reaching task
        # TODO: replace this codes later with spawn_dynamic_target func
        # Rotation = tf3d.euler.euler2mat(0.473038,-1.151761, -0.364322, 'sxyz')
        # Translation = [0.466187,  0.428941, 1.418610]
        # self.ee_target_pose = tf3d.affines.compose(Translation, Rotation, np.ones(3))
        self.ee_target_pose, target_pose = self.spawn_dynamic_reaching_goal('ball')
        self.goal = target_pose  # target pose, data type: geometry_msgs.msg.Pose

        # define the number of time step for every step know the time, then it can compute something
        # depend on the time
        self.contact_flag_released = True
        self.contact_flag = False
        self.tolerance = 1e-2  # reaching error threshold
        print("finish setup tiago reaching task env.")

    def _step(self, action):
        """
        Interact with env with policy learning with RL agent
        action: depend on the action space setting
        Return:  state, reward, and done status in this function
        The action command comes from an agent, which is an algorithm used for making decision
        """
        # Clip by veloctiy
        action = self._action_clip(action)  # should we used incremental command for a?

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # position clip
        action = np.clip(action, self.action_lower, self.action_upper).tolist()
        # positions_point = np.clip(action + np.array(joint_data.position),self.__joint_limits_lower, self.__joint_limits_upper)

        print("new action: {}".format(np.array(action)))

        try:
            # TODO: actually our action should consider the robot joint limit (including velocity limit)
            # TODO: add action prediction and corresponding terminate condition prediction before take excution
            # TODO: limit action (position translocation), keep every step have a very small moving.
            # we use joint position increment to send to robot

            # Control with action client
            self.arm_pos_control_client.wait_for_server()
            rospy.loginfo('connected to robot arm controller server')

            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.ctrl_joint_names
            g.trajectory.points = [JointTrajectoryPoint(positions=action, velocities=[0] * len(action), time_from_start=rospy.Duration(self.control_period))]
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
        state, abs_pos, joint_pos, joint_vel = self._get_obs()

        # TODO: add get the filtered force sensor data
        self.state = state
        self.joint_pos = joint_pos
        # current_jacobian = self._get_jacobian(joint_pos)

        # (needed by gym )done is the terminate condition. We should
        # return this value by considering the terminal condition.
        # TODO:  timestep stop set in max_episode_steps when register this env.
        # Another is using --nb-rollout-steps in running this env example?
        done = self.is_task_done(state, joint_vel)

        # TODO: add jacobian cost as part of reward, like GPS, which can avoid the robot explore the sigularity position.
        #  Also can add joint torque as part of reward, like GPS.
        # reward = max(0.0, 1.5 - 0.01*norm(np.array(end_pose_dist))**2)
        end_pose_dist = state[:7]
        distance = np.sqrt(np.sum(np.array(end_pose_dist[:3])**2) + ros_utils.distance_of_quaternion(end_pose_dist[3:7])**2)
        # TODO: we should add safety bounds information to reward, not only terminate condition,like some paper!
        reward = max(0.0, 2.0 - distance)
        print("=================step: %d, reward : %.3f, current dist: %.3f" % (self.time_step_index, reward, distance))

        # (needed by gym) we should return the state(or observation from state(function of state)), reward, and done status.
        # If the task completed, such as distance to target is d > = 0.001,
        # then return done= True, else return done = False. done depend on the terminal conditions partly.
        # NOTE: `reward` always depend on done and action, or state, so it always calculated finally.
        self.time_step_index += 1
        return np.array(self.state), reward, done, {}

    def _action_clip(self, action):
        """
        If action is beyond the current reaching goal, clip the action with the max velocity constraints
        :param action: original action value
        :return: clipped action value
        """
        action = np.array(action)
        safe_coef = 0.4
        pos_upper = np.array(self.joint_pos) + np.array(self.__joint_vel_limits_upper) * self.control_period * safe_coef
        pos_lower = np.array(self.joint_pos) + np.array(self.__joint_vel_limits_lower) * self.control_period * safe_coef
        if not (np.all(action <= pos_upper) and np.all(action >= pos_lower)):
            print("Invalid value, return clipped value")
            action = np.clip(action, pos_lower, pos_upper)
        return action

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
        ee_relative_pose, ee_abs_pos = self.get_ee_state()

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

        state = ee_relative_pose + force_data

        return state, ee_abs_pos, joint_pos, joint_vel

    def _reset(self, random=False):
        # we should stop our controllers firstly or send the initial joint angles to robot
        # using joint trajectory controller. Or the robot will back to its current position after unpause the simulation

        # reset arm position
        print("=====================================================================================================\n")
        rospy.loginfo('reset environment...')

        # reset robot first stage
        self.reset_world()
        self.ee_target_pose, self.goal = self.spawn_dynamic_reaching_goal('ball', random)
        self._virtual_reset_arm_config()

        # self._reset_hand_pose() # no hand, so deprecated
        time.sleep(0.2)

        print('===========================================reset done===============================================\n')

        # read data to observation

        # update `state`
        # state = self.discretize_observation(data, 5)
        state, abs_pos, joint_pos, joint_vel = self._get_obs()
        self.state = state
        self.joint_pos = joint_pos
        # self.time_step_index = 0

        # (needed by gym) return the initial observation or state
        return np.array(state)

    def get_ee_state(self):
        """Get end effector pose relative to the target.

        Return target pose relative to the target.
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

        ee_abs_pos = end_pose_affine[:3, 3].reshape(-1).tolist()

        # compute the relative pose to target pose (target frame relative the current frame)
        ee_pose_dist_affine = np.dot(inv(end_pose_affine), self.ee_target_pose)

        # translate the relative pose affine matrix to list (position, quaternion)
        ee_translation = ee_pose_dist_affine[:3, 3].reshape(-1).tolist()
        ee_quat = tf3d.quaternions.mat2quat(ee_pose_dist_affine[:3, :3]).tolist()
        ee_relative_pose = ee_translation + ee_quat

        # form the end-effector twist list
        ee_velocity = [ee_vel_msg.linear.x, ee_vel_msg.linear.y, ee_vel_msg.linear.z,
                       ee_vel_msg.angular.x, ee_vel_msg.angular.y, ee_vel_msg.angular.z]

        return ee_relative_pose, ee_abs_pos

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
            modelState.pose.position.x = x + np.random.sample() * 0.6 - 0.3
            modelState.pose.position.y = y
            modelState.pose.position.z = z + np.random.sample() * 0.6 - 0.3
        else:
            modelState.pose.position.x = x
            modelState.pose.position.y = y
            modelState.pose.position.z = z

        self.set_model_state(modelState)

        Rotation = tf3d.quaternions.quat2mat([modelState.pose.orientation.x, modelState.pose.orientation.y,
                                              modelState.pose.orientation.z, modelState.pose.orientation.w])
        Translation = [modelState.pose.position.x, modelState.pose.position.y, modelState.pose.position.z]
        target_pose = tf3d.affines.compose(Translation, Rotation, np.ones(3))

        hand_pose = modelState.pose  # different from the target pose
        hand_pose.position.x = x - 0.1
        return target_pose, hand_pose

    def is_task_done(self, state, joint_vel):

        # extract end pose distance from state
        end_pose_dist = state[:7]

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
            print('robot joint velocity exceed the joint velocity limit')
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
        done_condition1 = task_translation_error <= 0.03 and task_rotation_error > 0.1
        # task finished successfully
        done_condition2 = task_translation_error <= epsilon and task_rotation_error <= 200 * epsilon

        if done_condition1:
            print("Early termination for wrong policy")
            return True
        elif done_condition2:
            print("Task is over, goal reached")
            return True
        else:
            return False
