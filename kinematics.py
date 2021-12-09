"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

# from control_station import R2D
import numpy as np
from math import atan2, pi, acos, sqrt, cos, sin, degrees, radians, atan
# expm is a matrix exponential function
from scipy.linalg import expm
# from scipy.spatial.transform import Rotation as R
# from state_machine import D2R

# in degrees
Joint_limits = [ [-120, 120], [-100, 100], [-90, +111], [-120, +100], [-145, +145]]

def check_joint_limits(joint_angle, joint_index):
    if degrees(joint_angle) < Joint_limits[joint_index][0]:
        return radians(Joint_limits[joint_index][0])
    elif degrees(joint_angle) > Joint_limits[joint_index][1]:
        return radians(Joint_limits[joint_index][1])
    else:
        return joint_angle

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
    # euler_angles = get_euler_angles_from_T(T)
        
    # joint_positions = self.get_positions()

    phi = 0
    theta = 0
    dist = sqrt( x**2 + y**2 + z**2)
    if dist <=45.00/100 :
        psi = radians(90)
    else:
        psi = radians(45)
    euler_angles = [phi, theta, psi]

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

def pose_ik_elbow_down(pose, orientation, dh_params):
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

    # calculate angle base
    angle_base = atan2(-x, y)
    pitch = psi
    const_wrist_angle = radians(80)
    # print("angle base: ", degrees(angle_base))
    # print("pitch: ", degrees(pitch))

    x_c = x + l6*sin(angle_base)*cos(pitch)
    y_c = y - l6*cos(angle_base)*cos(pitch)
    z_c = z + l6*sin(pitch)
    z_c = z_c - (103.91/1000)

    # print("oc: ", x_c, y_c, z_c)

    pitch = psi
    # theta = 0
    phi = 0

    ## Use 2D planar IK solution to find other joints

    # Find theta 0 (slide 20)
    theta0 = angle_base
    theta0_w_pi = pi + theta0

    # let's try out a new way to calculate theta2 and theta1!
    # print('elbow_down: debug cos = ', (l2**2 + l1**2 - x_c**2 + y_c**2 - z_c**2) / (2*l1*l2))
    # theta2 = np.pi + atan2(200,50) - acos((l2**2 + l1**2 - x_c**2 - y_c**2 - z_c**2) / (2*l1*l2)) # in radians
    # theta2 = clamp(theta2)
    # alpha = atan2(l2*sin(theta2 - atan2(200,50)), l1 + l2*cos(theta2 - atan2(200,50)))
    # theta1 = atan2(200,50) + alpha - atan2(z_c, sqrt(x_c**2 + y_c**2))
    # theta1 = clamp(theta1)

    theta2 = acos((-l2**2 - l1**2 + x_c**2 + y_c**2 + z_c**2) / (-2*l1*l2)) - atan2(200,50)# in radians
    theta2 = clamp(theta2)
    alpha = atan2(l2*sin(theta2 - atan2(200, 50)), l1 + l2*cos(theta2 - atan2(200,50)))
    theta1 = atan2(200 , 50) + alpha - atan2(z_c, sqrt(x_c**2 + y_c**2))
    theta1 = clamp(theta1)


    # Make Shoulder smaller by 10 deg
    # theta1 -= radians(10) 

    # # Find 2 solutions for theta 2 (slide 9) --> is theta 2 in slides
    # theta2 = acos( ((x_c**2 + y_c**2 + z_c**2) - l1**2 - l2**2) / 2*l1*l2)   
    # # theta2 is elbow up, we get NEGATIVE theta2_calculated (we are in elbow up)
    # theta1 = atan2(z_c, sqrt(x_c**2 + y_c**2)) - atan2(l2*sin(theta2), l1+l2*cos(theta2)) 


    # theta2 = atan2(200,50) + theta2 
    # theta1 = atan2(200,50) - theta1 # handle offset

    # theta2 = clamp(theta2)
    # theta1 = clamp(theta1)
    # theta1 = np.pi / 2.0 - atan2(50, 200) - theta1

    
    # theta1 = theta1 + radians(12)

    # Now that we have theta 0, 1, and 2, we can find orientation of the wrist R_3^0 using forward kinematics (slide 24)

    ## Easy way to find theta4 & theta5
    theta_4 = theta1 - theta2 - pitch
    theta_4 = clamp(theta_4)
    theta_5 = theta
    theta_4_negative = -theta_4
    theta_4_negative = clamp(theta_4_negative)


    # Check for joint limits!
    theta0 = check_joint_limits(theta0, 0)
    theta1 = check_joint_limits(theta1, 1)
    theta2 = check_joint_limits(theta2, 2) 
    theta_4 = check_joint_limits(theta_4, 3)
    theta_5 = check_joint_limits(theta_5, 4)    

    # arrange results into matrix
    possible_joint_configs = np.array([
        [theta0, theta1, theta2, theta_4, theta_5],
        # [theta0_w_pi, theta1, theta2, theta_4_w_pi, theta_5_w_pi],
        [theta0, theta1, theta2, theta_4_negative, theta_5]
        # [theta0_w_pi, theta1, theta2, theta_4_negative_w_pi, theta_5_w_pi]
    ])

    # return 2 possible joint configurations (2 x 5 np.array)
    return possible_joint_configs



def pose_ik_elbow_up(pose, orientation, dh_params):
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

    # calculate angle base
    angle_base = atan2(-x, y)

    pitch = psi
    const_wrist_angle = radians(80)

    x_c = x + l6*sin(angle_base)*cos(pitch)
    y_c = y - l6*cos(angle_base)*cos(pitch)
    z_c = z + l6*sin(pitch)
    z_c = z_c - (103.91/1000)

    ## Use 2D planar IK solution to find other joints
    pitch = psi
    # theta = 0
    phi = 0

    # Find theta 0 (slide 20)
    theta0 = angle_base
    theta0_w_pi = pi + theta0

    # let's try out a new way to calculate theta2 and theta1!
    # print('l1', l1, 'l2', l2, 'x_c', x_c, 'y_c', y_c, 'z_c', z_c)
    # print('l1**2 ', l1**2, 'l2**2 ', l2**2, '2*l1*l2 ', 2*l1*l2)
    # print('xc**2 + yc**2 + zc**2 ', x_c**2 + y_c**2 + z_c**2)
    # print('elobw_up debug cos = ', (l2**2 + l1**2 - x_c**2 - y_c**2 - z_c**2) / (2*l1*l2) )
    # theta2 = radians(256) - acos((l2**2 + l1**2 - x_c**2 - y_c**2 - z_c**2) / (2*l1*l2)) # in radians
    # theta2 = clamp(theta2)
    theta2 = acos((-l2**2 - l1**2 + x_c**2 + y_c**2 + z_c**2) / (-2*l1*l2)) - atan2(200,50)# in radians
    theta2 = clamp(theta2)
    alpha = atan2(l2*sin(theta2 - atan2(200, 50)), l1 + l2*cos(theta2 - atan2(200,50)))
    theta1 = atan2(200 , 50) + alpha - atan2(z_c, sqrt(x_c**2 + y_c**2))
    theta1 = clamp(theta1)

    # Make Shoulder smaller by 10 deg
    # theta1 -= radians(10) 
    
    # print("theta1, theta2: ", theta1, theta2)

    # print('working cos = ', (x_c**2 + y_c**2 + z_c**2 - l1**2 - l2**2) / 2*l1*l2 ) 
    # theta2 = -acos( (x_c**2 + y_c**2 + z_c**2 - l1**2 - l2**2) / (2*l1*l2) )
    # theta1 = atan2(z_c, sqrt(x_c**2 + y_c**2)) - atan2(l2*sin(theta2), l1+l2*cos(theta2))

    # theta2 = atan2(200,50) + theta2 
    # # theta2 = theta2 + np.pi / 2.0
    # theta2 = theta2 + radians(12)   
    # theta2 = clamp(theta2)
    # theta1 = atan2(200,50) - theta1 # handle offset
    # theta1 = theta1 + radians(12)
    # theta1 = clamp(theta1)
    # theta1 = np.pi / 2.0 - atan2(50, 200) - theta1
    # Now that we have theta 0, 1, and 2, we can find orientation of the wrist R_3^0 using forward kinematics (slide 24)


    ## Easy way to find theta4 & theta5
    # Wrist Angle = Shoulder - Elbow - Pitch
    theta_4 = theta1 - theta2 - pitch
    theta_4 = clamp(theta_4)
    theta_5 = theta
    theta_4_negative = -theta_4
    theta_4_negative = clamp(theta_4_negative)


    # Check for joint limits!
    theta0 = check_joint_limits(theta0, 0)
    theta1 = check_joint_limits(theta1, 1)
    theta2 = check_joint_limits(theta2, 2) 
    theta_4 = check_joint_limits(theta_4, 3)
    theta_5 = check_joint_limits(theta_5, 4) 

    # arrange results into matrix
    possible_joint_configs = np.array([
        [theta0, theta1, theta2, theta_4, theta_5],
        # [theta0_w_pi, theta1, theta2, theta_4_w_pi, theta_5_w_pi],
        [theta0, theta1, theta2, theta_4_negative, theta_5]
        # [theta0_w_pi, theta1, theta2, theta_4_negative_w_pi, theta_5_w_pi]
    ])

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
    x, y, z, phi, theta, psi = pose    

    no_sol_config = np.array([
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0] ],
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0] ]
    ])

    # if sqrt(x**2 + y**2 + z**2) >= 40.00 /100:
        # return no_sol_config

    ########### TODO : determine Radius (measured from world (0, 0, 0) ##########
    if sqrt(x**2 + y**2) <= 15.00/100:
        z -= 0.065
    elif sqrt(x**2 + y**2) <= 20.00/100:
        z -= 0.03
    elif sqrt(x**2 + y**2) <= 25.00/100:
        z -= 0.015
    else:
        z -= 0.005

    # if sqrt(x**2 + y**2) <= 15.00/100:
    #     z -= 0.02
    # elif sqrt(x**2 + y**2) <= 20.00/100:
    #     z -= 0.01
    # elif sqrt(x**2 + y**2) <= 25.00/100:
    #     z -= 0.01
    # else:
    #     z -= 0.01

    # offset for (x, y)
    # 7.00
    # 8.00 worked
    if z >= 0.12:
        constant_offset = 9.00/100 # 3cm
        z -= 0.015
    elif z>= 0.9:
        constant_offset = 8.00/100
        z -= 0.015
    elif z>= 0.04:
        constant_offset = 7.00/100
        z -= 0.015
    else:
        constant_offset = 7.00/100
        
    # print("Before offset (x,y) ", x, y)
    base_angle = atan2(abs(x), abs(y))
    dx = constant_offset*sin(base_angle) 
    dy = constant_offset*cos(base_angle)


    if x <= 0.0:
        x = x + dx
    else:
        x = x - dx
    if y <= 0:
        y = y + dy
    else:
        y = y - dy

    # print("After offset (x,y) ", x, y)

    p = (x, y, z+ 0.0) # add 5cm to z
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





    ################ Below : NOT Used ##############
    # If z > 13 cm: set pitch to be 10 deg
    # if z >= 7.00/100:
    #     psi = radians(0)
    #     theta = 0
    # else:
    #     pass