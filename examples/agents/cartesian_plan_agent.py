# Plan arm to goal pose with traditional plan method

import gym


# ros related data structure
from geometry_msgs.msg import Twist, WrenchStamped, Pose, PoseStamped
from std_msgs.msg import Float64MultiArray, Header
# custom defined service call for moveit plan
from moveit_plan_service.srv import *

# ros related function packages
import rospy


env = gym.make('Tiago-v0')
env = env.unwrapped

rospy.wait_for_service('/gazebo/unpause_physics')
try:
    env.__unpause_physics()
except (rospy.ServiceException) as exc:
    print("/gazebo/unpause_physics service call failed:" + str(exc))

rospy.wait_for_service("/cartesian_arm_plan", 5)
cartesian_plan = rospy.ServiceProxy("/cartesian_arm_plan", CartesianArmPlan)
print("Plan to goal position with MoveIt planner")

header = Header()
header.frame_id = "base_footprint"
goal_pose = PoseStamped()
goal_pose.pose = env.goal
goal_pose.header = header

cartesian_plan(goal_pose)