# python related packages
import gym
import time
import numpy as np
from threading import Timer, Lock
from gym import error, spaces, utils
from gym.utils import seeding
from . import ros_general_utils as ros_utils  # custom user defined ros utils
from numpy.linalg import inv, norm
from gym_gazebo_ros.envs.tiago_robot.tiago_v0 import TiagoEnv

# ros related data structure
from geometry_msgs.msg import Twist, WrenchStamped, Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # Used for publishing UR joint angles.
from control_msgs.msg import *  # control with action interface
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


class TiagoPickEnv(TiagoEnv):
    def __init__(self):
        """
        Initialized Tiago robot Env
        """
        super(TiagoPickEnv, self).__init__()

