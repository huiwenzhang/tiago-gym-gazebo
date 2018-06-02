import gym
import rospy
#import roslaunch
import os
import signal
import subprocess
import time
from os import path
from std_srvs.srv import Empty
import random

from gazebo_msgs.srv import GetModelState, GetLinkState, SetModelConfiguration, DeleteModel, SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState, ContactState, ContactsState
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]

        with open("log.txt", "a") as myfile:
            myfile.write("export ROS_MASTER_URI=http://localhost:"+self.port + "\n")


        # Launch the simulation with the given launchfile name
        rospy.init_node('gym-gazebo', anonymous=True)
        print ("Make sure roslaunch is launched!")

        self.robot_name = rospy.get_param("/robot_name")  # steel/titanium
        rospy.wait_for_service("/gazebo/reset_world", 10.0)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        rospy.wait_for_service("/gazebo/get_model_state", 10.0)
        self.get_pose_srv = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState)

        # get the relative state
        rospy.wait_for_service("/gazebo/get_link_state", 10.0)
        self.get_link_pose_srv = rospy.ServiceProxy(
            "/gazebo/get_link_state", GetLinkState)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause_physics = rospy.ServiceProxy(
            "/gazebo/pause_physics", Empty)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause_physics = rospy.ServiceProxy(
            "/gazebo/unpause_physics", Empty)

        # ??
        rospy.wait_for_service("/controller_manager/switch_controller")
        self.switch_ctrl = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)

        rospy.wait_for_service("/gazebo/set_model_configuration")
        self.set_model = rospy.ServiceProxy(
            "/gazebo/set_model_configuration", SetModelConfiguration)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)

        rospy.wait_for_service("/gazebo/delete_model")
        self.delete_model = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel)
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel)

        print("ROS services setup done...")

        self.gzclient_pid = 0

    def _step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError


    def _reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0

    def _close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
