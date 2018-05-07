import numpy as np
import rospy

from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_kinematics import KDLKinematics
from assembler_bridge.srv import  AssemblerJacobian


def get_position(tf, target, source, time):
    """
    Utility function that uses tf to return the position of target
    relative to source at time
    tf: Object that implements TransformListener
    target: Valid label corresponding to target link
    source: Valid label corresponding to source link
    time: Time given in TF's time structure of secs and nsecs
    """
    # Calculate the quaternion data for the relative position
    # between the target and source.
    translation, rot = tf.lookupTransform(target, source, time)

    # Get rotation and translation matrix from the quaternion data.
    # The top left 3x3 section is a rotation matrix.
    # The far right column is a translation vector with 1 at the bottom.
    # The bottom row is [0 0 0 1].
    transform = np.asmatrix(tf.fromTranslationRotation(translation, rot))

    # Get position relative to source by multiplying the rotation by 
    # the translation. The -1 is for robot matching sign conventions.

    # TODO: why not directly to translation? comment by L
    position = -1 * (transform[:3, 3].T * transform[:3, :3])

    # Convert from np.matrix to np.array
    position = np.asarray(position)[0][:]

    return position

def approx_equal(a, b, threshold=1e-5):
    """
    Return whether two numbers are equal within an absolute threshold.
    Returns:
        True if a and b are equal within threshold.
    """
    return np.all(np.abs(a - b) < threshold)


def get_ee_points_position(offsets, ee_pos, ee_rot):
    """
    Helper method for computing the end effector points given a
    position, rotation matrix, and offsets for each of the ee points.

    Args:
        offsets: N x 3 array where N is the number of points.
        ee_pos: 1 x 3 array of the end effector position.
        ee_rot: 3 x 3 rotation matrix of the end effector.
    Returns:
        3 x N array of end effector points.
    """
    return ee_rot.dot(offsets.T) + ee_pos.T


def get_jacobians(q, robot, base_link, end_link):
    jacobian = np.zeros((6,6))
    rospy.wait_for_service('get_jacobian_srv')
    try:
        get_jacobian = rospy.ServiceProxy('get_jacobian_srv', AssemblerJacobian)
        resp = get_jacobian(q)

        for i in range(6):
            for j in range(6):
                jacobian[i][j] = resp.jacobian[i*6+j]

        # return jacobian matrix for current joint state q
        return jacobian
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)




def distance_of_quaternion(delta_q):
    """
    Args:
        delta_q: delta quaternion from past quaternion to current quaternion (q*q^bar)

    Return: 
        the distance of the quarternions
    """
    real_q = np.array(delta_q[0])
    u_q = np.array(delta_q[1:])
    if np.linalg.norm(u_q) != 0.0:
        log_q = np.arccos(real_q)* u_q/np.linalg.norm(u_q)
    else:
        log_q = np.zeros(3)

    if (real_q != -1) or (np.linalg.norm(u_q) != 0.0):
        d = np.linalg.norm(log_q)
    else:
        d = np.pi 
    
    return d


