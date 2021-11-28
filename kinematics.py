"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
from math import atan2, pi, acos, sqrt, cos, sin
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    pass


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """

    # T : 4x4
    Rotation = T[0:3, 0:3]
    # print("Rotation: ", Rotation)

    dcm = R.from_dcm(Rotation)
    # print("r: ", r) 
    
    euler_angles = dcm.as_euler('zyz', degrees=False) # False: use radian
    # print("r ", r)

    return euler_angles # a 3x3 matrix

    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def pose_ik_elbow_up(pose, orientation, dh_params):
    x,y,z = pose
    theta, phi, psi = orientation

    # Find R using euler angles (slide 25)
    R = np.array([
        [cos(phi)*cos(theta)*cos(psi) - sin(phi)*sin(psi), -cos(phi)*cos(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*sin(theta)],
        [sin(phi)*cos(theta)*cos(psi) + cos(phi)*sin(psi), -sin(phi)*cos(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*sin(theta)],
        [-sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)]
    ])
    # o is just the x, y, z position of the goal from the start (slide 17)
    o = np.array([
        [x],
        [y],
        [z]
    ])
    print('o = ', o)

    # Read in joint lengths (l1, l2, ... l6)
    l1 = 205.73 / 1000 # might have to change this based on the slight offset from joint 1 to 2
    l2 = 200 / 1000 
    # Define l6 -> distance from wrist to end effector
    l6 = 174.15 / 1000

    # Find o_c (slide 19 of IK lecture)
    print(R[:, 2])
    o_c = np.add(o, -l6 * R[:, 2].reshape(-1, 1)) # 3rd column
    x_c = o_c[0] 
    print('o_c = ', o_c)
    y_c = o_c[1] 
    z_c = o_c[2]

    ## Use 2D planar IK solution to find other joints

    # Find theta 0 (slide 20)
    theta1_1 =  atan2(y_c, x_c)
    theta1_2 = pi + atan2(y_c, x_c)

    # Find 2 solutions for theta 3 (slide 9) --> is theta 2 in slides
    theta3_1 = acos(((x_c**2 + y_c**2) - l1**2 - l2**2) / 2*l1*l2)
    theta3_2 = -theta3_1

    # theta2_1 is elbow up
    # theta2_2 is elbow down

    # Find theta 2 based on theta 2 (slide 9) --> is theta 1 in slides
    theta2_1 = atan2(y_c, x_c) - atan2(l2*sin(theta3_1), l1+l2*cos(theta3_1))
    theta2_2 = atan2(y_c, x_c) - atan2(l2*sin(theta3_2), l1+l2*cos(theta3_2))

    # Now that we have theta 1, 2, and 3, we can find orientation of the wrist R_3^0 using forward kinematics (slide 24)

    # Let's do for elbow up first
    c1 = cos(theta1_1)
    c23 = cos(theta2_1 + theta3_1)
    s23 = sin(theta2_1 + theta3_1)
    s1 = sin(theta1_1)
    # Calculate R_3^0 (slide 24)
    R_3_0 = np.array([
        [c1*c23, -c1*s23, s1],
        [s1*c23, -s1*s23, -c1],
        [s23, c23, 0]
    ])
    # Calculate R_6^3
    R_6_3 = np.matmul(np.transpose(R_3_0), R)

    # Get theta 4, 5, and 6 from R_6_3
    theta_4_1 = atan2(R_6_3[1, 2], R_6_3[0, 2])
    theta_5_1 = atan2(sqrt(1 - R_6_3[2, 2]**2), R_6_3[2, 2])
    theta_5_2 = atan2(-sqrt(1 - R_6_3[2, 2]**2), R_6_3[2, 2])
    theta_6_1 = atan2(R_6_3[2, 1], -R_6_3[2, 0])

    # arrange results into matrix
    possible_joint_configs = np.array([
        [theta1_1, theta2_1, theta3_1, theta_4_1, theta_5_1, theta_6_1],
        [theta1_2, theta2_2, theta3_2, theta_4_1, theta_5_2, theta_6_1]
    ])
    
    # return 2 possible joint configurations (2 x 5 np.array)
    return possible_joint_configs

def pose_ik_elbow_down(pose, orientation, dh_params):


    # return 2 possible joint configurations
    possible_joint_configs = np.zeros([2, 6])
    return possible_joint_configs

def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    
    joint_configs = np.zeros(shape=(4,5)) # 4x5


    # 
    x, y, z, phi, theta, psi = pose
    p = (x, y, z)
    ori = (theta, phi, psi)
    el_up = pose_ik_elbow_up(p,
                             ori,
                             dh_params= dh_params)
    print('joints up = ', el_up)
    print('done joints')
    # return 2x5 array

    el_down = pose_ik_elbow_down(p,
                             ori,
                             dh_params= dh_params)
    # return 2x5 array
    # print(el_down.shape)

    # joint_configs = np.array([el_up,  el_down])
    joint_configs = np.stack((el_up, el_down))

    return joint_configs