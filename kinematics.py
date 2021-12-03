"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
from math import atan2, pi, acos, sqrt, cos, sin, degrees, radians, atan
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

    # Load DH for each frame (pass in control_station.py), use 
    # self.dh_params (a numpy array 5*4)
    # each row: a, alpha, d, theta for each joint
    # joint: 0 ~ 4

    # a, alpha: w.r.t the old one (diff between Z axes)
    # d, theta: w.r.t the new one (diff between X axes)

    # Get current positions
    current_positions = joint_angles
    # print("cur positions: ", current_positions)
    # first angle is theta 1

    # H01 => theta = 1 
    # H = np.array(shape=(5, 4,4 ))
    # Compute H01, H12, H23, H34, H4e ( H01: From 0 to 1 )
    Hs = []
    # print("dh: ", self.dh_params)
    for i in range(len(dh_params)): # 6x4 iterate for 5 joints
        
        a = dh_params[i][0]
        # convert a from mm to m
        a = a / 1000.0
        
        alpha = dh_params[i][1]
        # convert alpha from degrees to radians
        alpha = radians(alpha)
        
        d = dh_params[i][2]
        # convert d from mm to m
        d = d / 1000.0
        
        theta = dh_params[i][3]
        if i == 0: # World to base: use cur_position[0]
            theta = degrees(current_positions[0])
        elif i==1: # Base to Sholder: constant 90 degree
            theta = 90 
        elif i==2: # Sholder to elbow
            theta =  degrees(current_positions[1]) - (90 - degrees(np.arctan(1/4))) 
        elif i==3:
            theta =  degrees(current_positions[2]) - (90 - degrees(np.arctan(1/4))) 
        elif i==4: 
            theta = degrees(current_positions[3]) + 90                 
        elif i==5: # Wrist to ee : constant 0 degree
            pass

        # convert theta from degrees to radians
        theta = radians(theta)

        # print("i ", i, " a, alpha, d, theta, ", a," ", alpha, " ", d , " ", theta)

        # Make H 0-> 1 4x4 HTM
        H = np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha),  a*cos(theta)],
                        [sin(theta),  cos(theta)*cos(alpha),  -cos(theta)*sin(alpha), a*sin(theta)],
                        [         0,             sin(alpha),    cos(alpha),        d],
                        [         0,                      0,             0,        1]])
        Hs.append(H)
    # while True:
    #     pass

    HW0 = Hs[0]
    H01 = Hs[1]
    H12 = Hs[2]
    H23 = Hs[3]
    H34 = Hs[4]
    H4e = Hs[5]
    # numpy.array so we can use @ !
    
    Hwe =  np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(HW0,H01), H12), H23), H34), H4e)

    # H3e = np.matmul(H34, H4e)

    return Hwe


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
    Rotation = T[0:3, 0:3] # R06
    # print('original R = ', Rotation)
    # print()

    # FK slides 25 
    # Case 1 -> r13, r23 != 0
    r13 = Rotation[0, 2]
    r23 = Rotation[1, 2]
    r33 = Rotation[2, 2]
    r21 = Rotation[1, 0]
    r11 = Rotation[0, 0]
    r32 = Rotation[2, 1]
    r31 = Rotation[2, 0]
    cos_theta = Rotation[2,2]
    # Case 2
    if abs(r13) < 1e-6 and abs(r23) < 1e-6 and abs(r33) == 1:
        sin_theta = 0
        psi = atan2(r21, r11) / 2.0
        phi = atan2(r21, r11) / 2.0
        # theta is calculated below
    else: # Case 1
        if cos_theta >= 0:
            sin_theta = -sqrt(1 - cos_theta*cos_theta)
        else:
            sin_theta = sqrt(1 - cos_theta*cos_theta)

        if sin_theta >= 0:
            phi = atan2(r23, r13)
            psi = atan2(r32, -r31)
        else:
            phi = atan2(-r23, -r13)
            psi = atan2(-r32, r31)   

    theta = atan2(sin_theta, cos_theta)    

    # calculate R again
    newR = np.array([
        [cos(phi)*cos(theta)*cos(psi) - sin(phi)*sin(psi), -cos(phi)*cos(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*sin(theta)],
        [sin(phi)*cos(theta)*cos(psi) + cos(phi)*sin(psi), -sin(phi)*cos(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*sin(theta)],
        [-sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)]
    ])

    # print('new R = ', newR)
    # print()

    # get 5 joints positions
    # joint_positions = self.rxarm.get_positions()


    # print("Rotation: ", Rotation)
    # print("Rotation type: ", type(Rotation))
    # dcm = R.from_dcm(Rotation)
    # print("r: ", r) 
    # r =  R.from_rotvec(Rotation)
    # euler_angles = dcm.as_euler('zyz', degrees=False) # False: use radian
 

    phi = atan2(r23, r13)
    theta = atan2( sqrt(1-r33**2), r33)
    psi = atan2(r32, -r23)

    euler_angles = np.array([phi, theta, psi])

    return euler_angles # a 3x1 matrix

    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    euler_angles = get_euler_angles_from_T(T)
        
    # joint_positions = self.get_positions()
    # phi = joint_positions[0] + np.pi/2
    # theta = joint_positions[4]
    # psi = joint_positions[1] - joint_positions[2] - joint_positions[3]


    pose = [x, y, z, euler_angles[0], euler_angles[1], euler_angles[2]]

    return pose


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


def wrap_to_pi(angle):
    # restrict between [-pi , pi]
    if angle > pi:
        while angle > pi:
            angle -= 2*pi
    if angle < -pi:
        while angle < -pi:
            angle += 2*pi
            
    return angle

def pose_ik_elbow_up(pose, orientation, dh_params):
    x,y,z = pose
    phi, theta, psi = orientation



    # Find R using euler angles (slide 25)
    R = np.array([
        [cos(phi)*cos(theta)*cos(psi) - sin(phi)*sin(psi), -cos(phi)*cos(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*sin(theta)],
        [sin(phi)*cos(theta)*cos(psi) + cos(phi)*sin(psi), -sin(phi)*cos(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*sin(theta)],
        [-sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)] # NOte: The 3rd column Theta changes to psi+pi/2
    ])
    # o is just the x, y, z position of the goal from the start (slide 17)
    o = np.array([
        [x],
        [y],
        [z]
    ])
    # print('o = ', o)

    # Read in joint lengths (l1, l2, ... l6)
    l1 = 205.73 / 1000 # might have to change this based on the slight offset from joint 1 to 2
    l2 = 200.00 / 1000 
    # Define l6 -> distance from wrist to end effector
    l6 = 174.15 / 1000

    # Find o_c (slide 19 of IK lecture)
    # print(R[:, 2])

    # print("R 3rd column ", R[:,2])
    o_c = np.add(o, -l6 * R[:, 2].reshape(-1, 1)) # 3rd column

    # print("oc: ",o_c)
    # x_c = o_c[0].item(0)
    # print('o_c = ', o_c)
    # y_c = o_c[1].item(0)
    # z_c = o_c[2].item(0)

    # calculate angle base
    angle_base = atan2(-x, y)

    pitch = psi

    # x_c = x - l6*cos(phi)
    x_c = x + l6*sin(angle_base)*cos(pitch)
    # y_c = y - l6*sin(phi)
    y_c = y - l6*cos(angle_base)*cos(pitch)
    # z_c = z + l6*sin(psi) # psi is pitch
    z_c = z + l6*sin(pitch)
    z_c = z_c - (103.91/1000)

    print("psi: ",psi)
    print("sin(psi) ", sin(psi))
    print('xc = ', x_c)
    print('yc = ', y_c)
    print('zc = ', z_c)        

    pitch = psi
    theta = 0
    phi = 0

    ## Use 2D planar IK solution to find other joints

    # Find theta 0 (slide 20)
    theta0 = atan2(y_c, x_c) - pi/2
    theta0_w_pi = pi + theta0

    # print("x_c type: ",type(x_c))
    # print("x_c: ", x_c)
    # print("x_c shape: ", x_c.shape)
    # print("sqrt: ",sqrt(x_c**2 + y_c**2) )
    # print(" l1**2 - l2**2 ", l1**2 - l2**2)
    # print("l1 ", l1)
    # print("l2 ", l2)
    # print("type: ", type(l1), type(l2))
    
    # print(" 2*l1*l2 ", 2*l1*l2)
    # print("numerator: ",((sqrt(x_c**2 + y_c**2) + z_c**2) - l1**2 - l2**2))

    # Find 2 solutions for theta 2 (slide 9) --> is theta 2 in slides
    theta2 = acos( ((sqrt(x_c**2 + y_c**2) + z_c**2) - l1**2 - l2**2) / 2*l1*l2)
    # theta2 is elbow up, we get NEGATIVE theta2_calculated (we are in elbow up)
    # theta2 = atan2(200,50) - theta2 # theta2 would be smaller than atan2(4) 
    theta2 = pi/2 - theta2
    # Find theta 1 based on theta 2 (slide 9) --> is theta 1 in slides
    # print("z_c, x_c, y_c ", z_c, x_c, y_c)
    # print("term1 ",atan2(z_c, sqrt(x_c**2 + y_c**2)))
    # print("theta2 ", theta2)
    # print("l2*sin2 l1+l2*cos2 ", l2*sin(theta2), l1+l2*cos(theta2))
    # print("term2 ",atan2(l2*sin(theta2), l1+l2*cos(theta2))  )

    theta1 = atan2(z_c, sqrt(x_c**2 + y_c**2)) - atan2(l2*sin(theta2), l1+l2*cos(theta2)) 

    ##### Problem Here ! #####
    theta1 = atan2(200,50) + theta1 # handle offset

    # Now that we have theta 0, 1, and 2, we can find orientation of the wrist R_3^0 using forward kinematics (slide 24)

    # # Let's do for elbow up first
    # c1 = cos(theta0)
    # c1_w_pi = cos(theta0_w_pi)
    # c23 = cos(theta1 + theta2)
    # s23 = sin(theta1 + theta2)
    # s1 = sin(theta0)
    # s1_w_pi = sin(theta0_w_pi)
    # # Calculate R_3^0 (slide 24)
    # R_3_0 = np.array([
    #     [c1*c23, -c1*s23, s1],
    #     [s1*c23, -s1*s23, -c1],
    #     [s23, c23, 0]
    # ])
    # # Calculate with pi offset
    # R_3_0_w_pi = np.array([
    #     [c1_w_pi*c23, -c1_w_pi*s23, s1_w_pi],
    #     [s1_w_pi*c23, -s1_w_pi*s23, -c1_w_pi],
    #     [s23, c23, 0]
    # ])
    # # Calculate R_6^3
    # R_6_3 = np.matmul(np.transpose(R_3_0), R)
    # R_6_3_w_pi = np.matmul(np.transpose(R_3_0_w_pi), R)

    # # Get theta 4, 5 from R_6_3 and R_6_3_w_pi
    # # theta_3_1 = atan2(R_6_3[1, 2], R_6_3[0, 2])
    # theta_4 = atan2(sqrt(1 - R_6_3[2, 2]**2), R_6_3[2, 2])
    # theta_4_negative = atan2(-sqrt(1 - R_6_3[2, 2]**2), R_6_3[2, 2])
    # theta_4_w_pi = atan2(sqrt(1 - R_6_3_w_pi[2, 2]**2), R_6_3_w_pi[2, 2])
    # theta_4_negative_w_pi = atan2(-sqrt(1 - R_6_3_w_pi[2, 2]**2), R_6_3_w_pi[2, 2])
    # theta_5 = atan2(R_6_3[2, 1], -R_6_3[2, 0])
    # theta_5_w_pi = atan2(R_6_3_w_pi[2, 1], -R_6_3_w_pi[2, 0])

    ## Easy way to find theta4 & theta5
    theta_4 = theta1 - theta2 - pitch
    theta_5 = 0 
    theta_4_negative = -theta_4

    # arrange results into matrix
    possible_joint_configs = np.array([
        [theta0, theta1, theta2, theta_4, theta_5],
        # [theta0_w_pi, theta1, theta2, theta_4_w_pi, theta_5_w_pi],
        [theta0, theta1, theta2, theta_4_negative, theta_5]
        # [theta0_w_pi, theta1, theta2, theta_4_negative_w_pi, theta_5_w_pi]
    ])
    
    for configs in possible_joint_configs:
        for i in range(len(configs)):
            configs[i] = wrap_to_pi(configs[i])

    # return 2 possible joint configurations (2 x 5 np.array)
    return possible_joint_configs




def pose_ik_elbow_down(pose, orientation, dh_params):
    x,y,z = pose
    phi, theta, psi = orientation

    # Find R using euler angles (slide 25) R06
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
    # print('o = ', o)

    # Read in joint lengths (l1, l2, ... l6)
    l1 = 205.73 / 1000 # might have to change this based on the slight offset from joint 1 to 2 
    l2 = 200.00 / 1000 
    # Define l6 -> distance from wrist to end effector
    l6 = 174.15 / 1000

    # Find o_c (slide 19 of IK lecture)
    # print(R[:, 2])
    # o_c = np.add(o, -l6 * R[:, 2].reshape(-1, 1)) # 3rd column

    # calculate angle base
    angle_base = atan2(-x, y)

    pitch = psi

    # x_c = x - l6*cos(phi)
    x_c = x + l6*sin(angle_base)*cos(pitch)
    # y_c = y - l6*sin(phi)
    y_c = y - l6*cos(angle_base)*cos(pitch)
    # z_c = z + l6*sin(psi) # psi is pitch
    z_c = z + l6*sin(pitch)
    z_c = z_c - (103.91/1000)

    # x_c = x - l6*cos(phi)
    # y_c = y - l6*sin(phi)
    # z_c = z + l6*sin(psi) # good one
    # z_c = z_c - (103.91/1000)

    ## Use 2D planar IK solution to find other joints
    pitch = psi
    theta = 0
    phi = 0


    # Find theta 0 (slide 20)
    theta0 = atan2(y_c, x_c) - pi/2
    theta0_w_pi = pi + theta0

    # Find 2 solutions for theta 2 (slide 9) --> is theta 2 in slides
    # print("(sqrt(x_c**2 + y_c**2) + z_c**2) ", (sqrt(x_c**2 + y_c**2) + z_c**2))
    # print("l1**2 , l2**2 ",l1**2 , l2**2 )
    # print("numerator : ", ((sqrt(x_c**2 + y_c**2) + z_c**2) - l1**2 - l2**2))
    # print("denom: ", 2*l1*l2)
    theta2 = -acos( ((sqrt(x_c**2 + y_c**2) + z_c**2) - l1**2 - l2**2) / 2*l1*l2)
    # theta2 is elbow down, we get POSITIVE theta2_calculated (we are in elbow down)
    # theta2 = atan2(200,50) +- theta2 # theta2 would be greater than atan2(4)
    
    # Find theta 1 based on theta 2 (slide 9) --> is theta 1 in slides
    theta1 = atan2(z_c, sqrt(x_c**2 + y_c**2)) - atan2(l2*sin(theta2), l1+l2*cos(theta2))
    
    theta2 = pi/2 - theta2     
    theta1 = pi/2 - theta1  # handle offset

    # Now that we have theta 0, 1, and 2, we can find orientation of the wrist R_3^0 using forward kinematics (slide 24)

    # # Let's do for elbow up first
    # c1 = cos(theta0)
    # c1_w_pi = cos(theta0_w_pi)
    # c23 = cos(theta1 + theta2)
    # s23 = sin(theta1 + theta2)
    # s1 = sin(theta0)
    # s1_w_pi = sin(theta0_w_pi)
    # # Calculate R_3^0 (slide 24)
    # R_3_0 = np.array([
    #     [c1*c23, -c1*s23, s1],
    #     [s1*c23, -s1*s23, -c1],
    #     [s23, c23, 0]
    # ])
    # # Calculate with pi offset
    # R_3_0_w_pi = np.array([
    #     [c1_w_pi*c23, -c1_w_pi*s23, s1_w_pi],
    #     [s1_w_pi*c23, -s1_w_pi*s23, -c1_w_pi],
    #     [s23, c23, 0]
    # ])
    # # Calculate R_6^3
    # R_6_3 = np.matmul(np.transpose(R_3_0), R)
    # R_6_3_w_pi = np.matmul(np.transpose(R_3_0_w_pi), R)

    # # Get theta 4, 5 from R_6_3 and R_6_3_w_pi
    # # theta_3_1 = atan2(R_6_3[1, 2], R_6_3[0, 2])
    # theta_4 = atan2(sqrt(1 - R_6_3[2, 2]**2), R_6_3[2, 2])
    # theta_4_negative = atan2(-sqrt(1 - R_6_3[2, 2]**2), R_6_3[2, 2])
    # theta_4_w_pi = atan2(sqrt(1 - R_6_3_w_pi[2, 2]**2), R_6_3_w_pi[2, 2])
    # theta_4_negative_w_pi = atan2(-sqrt(1 - R_6_3_w_pi[2, 2]**2), R_6_3_w_pi[2, 2])
    # theta_5 = atan2(R_6_3[2, 1], -R_6_3[2, 0])
    # theta_5_w_pi = atan2(R_6_3_w_pi[2, 1], -R_6_3_w_pi[2, 0])


    ## Easy way to find theta4 & theta5
    theta_4 = theta1 - theta2 - pitch
    theta_5 = 0 
    theta_4_negative = -theta_4


    # arrange results into matrix
    possible_joint_configs = np.array([
        [theta0, theta1, theta2, theta_4, theta_5],
        # [theta0_w_pi, theta1, theta2, theta_4_w_pi, theta_5_w_pi],
        [theta0, theta1, theta2, theta_4_negative, theta_5]
        # [theta0_w_pi, theta1, theta2, theta_4_negative_w_pi, theta_5_w_pi]
    ])
    
    for configs in possible_joint_configs:
        for i in range(len(configs)):
            # print("theta ", theta)
            configs[i] = wrap_to_pi(configs[i])
            # print("after wrap_to_pi ", theta)

    # return 2 possible joint configurations (2 x 5 np.array)
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
    ori = (phi, theta, psi)
    el_up = pose_ik_elbow_up(p,
                             ori,
                             dh_params= dh_params)
    # print('joints up = ', el_up)
    # print('done joints')
    # return 2x5 array

    el_down = pose_ik_elbow_down(p,
                             ori,
                             dh_params= dh_params)
    # return 2x5 array
    # print(el_down.shape)

    # joint_configs = np.array([el_up,  el_down])
    joint_configs = np.stack((el_up, el_down))

    return joint_configs