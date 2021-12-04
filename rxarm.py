"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
from sys import path_importer_cache
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, Joint_limits, get_pose_from_T, get_euler_angles_from_T, IK_geometric
import time
import csv
from builtins import super
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
from interbotix_robot_arm import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
from config_parse import *
from sensor_msgs.msg import JointState
import rospy


from math import cos, sin, radians, degrees, fabs

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixRobot):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_name="rx200", mrd=mrd)
        self.joint_names = self.resp.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None

        # DH Params
        self.dh_params = []
        # self.dh_params = np.array([])
        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            print("df file: ", dh_config_file)
            self.dh_params = RXArm.parse_dh_param_file(self)


            
        #POX params
        self.M_matrix = []
        self.S_list = []

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        rospy.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.set_gripper_pressure(1.0)
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.open_gripper()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_off(self.joint_names)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_on(self.joint_names)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


#   @_ensure_initialized

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.

        @return     The EE pose as [x, y, z, phi, Theta, Psi] or as needed.
        """
        # 3.2 TODO: Use DH table peform Forward Kinematics Transformation

        # Load DH for each frame (pass in control_station.py), use 
        # self.dh_params (a numpy array 5*4)
        # each row: a, alpha, d, theta for each joint
        # joint: 0 ~ 4

        # a, alpha: w.r.t the old one (diff between Z axes)
        # d, theta: w.r.t the new one (diff between X axes)

        # Get current positions
        current_positions = self.get_positions()
        # print("cur positions: ", current_positions)
        # first angle is theta 1

        # H01 => theta = 1 
        # H = np.array(shape=(5, 4,4 ))
        # Compute H01, H12, H23, H34, H4e ( H01: From 0 to 1 )
        Hs = []
        # print("dh: ", self.dh_params)
        for i in range(len(self.dh_params)): # 6x4 iterate for 5 joints
            
            a = self.dh_params[i][0]
            # convert a from mm to m
            a = a / 1000.0
            
            alpha = self.dh_params[i][1]
            # convert alpha from degrees to radians
            alpha = radians(alpha)
            
            d = self.dh_params[i][2]
            # convert d from mm to m
            d = d / 1000.0
            
            theta = self.dh_params[i][3]
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

        # print("Final H world to ee: ", Hwe)
        Translation = Hwe[0:3, 3]
        x = Translation[0]
        y = Translation[1]
        z = Translation[2]
        # print("x, y, z", x, y, z)


        H2e = np.matmul( np.matmul(H23,H34), H4e)
        # euler_angles = get_euler_angles_from_T(Hwe)
        # euler_angles = get_euler_angles_from_T(Hwe)
        # phi = euler_angles[0]
        # theta = euler_angles[1]
        # psi = euler_angles[2]
        
        joint_positions = self.get_positions()
        phi = joint_positions[0] + np.pi/2
        theta = joint_positions[4]
        # Pitch = psi
        # Psi = Shoulder - Elbow - Wrist Angle
        psi = joint_positions[1] - joint_positions[2] - joint_positions[3]

        pose = [x, y, z, phi, theta, psi]

        # Pass FK pose into IK_geometry
        # joint_calculated = IK_geometric(self.dh_params, pose)
        # for i in range(len(joint_calculated)):
        #     joint_calculated[i] *= R2D

        # print("joint from IK: ", joint_calculated)

        return pose
        # return [0, 0, 0, 0]


    def find_best_soluton(self, joint_angles, elbow_status, target_z):
        # print("joint_angles: ", joint_angles)

        solutions = joint_angles[elbow_status][:][:]
        # print("solutions ", solutions)
        for solution in solutions:
            # print("solultion: ", solution)
            link = 0 # we didn't use it
            T = FK_dh(self.dh_params, solution, link)
            ee_pos = get_pose_from_T(T)
            # if z > 1cm and |x| > 45 cm
            x, y, z, phi, theta, psi = ee_pos
            print("x, z: ", x, z)
            if z > 0.00/100 and x<= 45.00/100 and x >= -45.00/100 and solution[3]<=radians(0):
                return solution
                # if elbow_status==0: # elbow up, discard theta3>0 solution
                #     if solution[3] <= 0:
                #         return solution
                # else:
                #     if solution[3] >= 0:
                #         return solution
        # if NO sol'z > 1cm: 
        print("No valid solution")
        return np.array([0, 0, 0, 0, 0])




    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        # define PID values
        waist_PID = [640, 0, 3600]
        shoulder_PID = [700, 100, 0]
        elbow_PID = [700, 80, 0]
        wrist_angle_PID = [800, 0, 0]
        wrist_rotate_PID = [640, 0, 3600]
        gripper_PID = [640, 0, 3600]
        pid_gains = [waist_PID, shoulder_PID, elbow_PID, wrist_angle_PID, wrist_rotate_PID, gripper_PID]
        # assign PID gains
        i = 0
        for joint in self.rxarm.joint_names:
            self.rxarm.set_joint_position_pid_params(joint, pid_gains[i])
            i = i + 1
            print("joint ", joint, "pid: ", self.rxarm.get_joint_position_pid_params(joint))
        rospy.Subscriber('/rx200/joint_states', JointState, self.callback)
        rospy.sleep(0.5)



    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        # print out IK solutions for debugging
        # joints_debug = IK_geometric(self.rxarm.dh_params, self.rxarm.get_ee_pose())
        # print('IK joints debug = ', joints_debug * R2D)
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:

            rospy.spin()



if __name__ == '__main__':
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.close_gripper()
        rxarm.go_to_home_pose()
        rxarm.open_gripper()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
