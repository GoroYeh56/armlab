"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
from camera import Block
from kinematics import FK_dh, FK_pox, get_pose_from_T, get_euler_angles_from_T, IK_geometric
import time
import numpy as np
import rospy
import cv2
from math import radians, sqrt, degrees, atan2
from copy import deepcopy


green_up_config = np.array([-45, 6, 8.2, -44.56, 0.53]) 
green_far_config = np.array([-47.37, 49.66, 4.92, 43.59, -2.11]) 
drop_green_config = np.array([-29.36, 9.58, -17.31, -52.56, 2.2])

purple_up_config= np.array([-45, 6, 8.2, -44.56, 0.53])
purple_config= np.array([-46.41, 49.92, 11.43, 25.58, -9.4])
drop_purple_config = np.array([-42.98, 29, 17.14, 61.96, -8.88])


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


for i in range(5):
    green_up_config[i] = clamp(radians(green_up_config[i]))
    green_far_config[i] = clamp(radians(green_far_config[i]))
    drop_green_config[i] = clamp(radians(drop_green_config[i]))

    purple_up_config[i] = clamp(radians(purple_up_config[i]))
    purple_config[i] = clamp(radians(purple_config[i]))
    drop_purple_config[i] = clamp(radians(drop_purple_config[i]))


D2R = np.pi / 180.0
R2D = 180.0 / np.pi

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = []
        self.replay_buffer = [] # a buffer to check whether it is a "intermediate waypoint":0, "should close gripper":-1, "should open gripper":1

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "teach":
            self.teach()
        
        if self.next_state == "pick_place":
            self.pick_place()

        # if self.next_state == "line_em_up":
            # self.pick_place()



    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()
        self.waypoints = []
        self.replay_buffer = []

    def teach(self):
        self.status_message = "Enter Teach state..."
        self.current_state = "teach"
        self.rxarm.disable_torque()


    ### @brief Records the current joint positions to global var
    def record_joint_angles(self):
        self.waypoints.append(self.rxarm.get_positions())
        print("Add waypoint: ", self.waypoints[-1])
        self.replay_buffer.append(0)

    ### @brief Records position for opening gripper
    def record_open(self):
        # add waypoint
        self.waypoints.append(self.rxarm.get_positions())
        self.replay_buffer.append(1)
        print('open gripper...')

    ### @brief Records position for closing gripper
    def record_closed(self):
        # add waypoint
        self.waypoints.append(self.rxarm.get_positions())
        self.replay_buffer.append(-1)
        print('close gripper...')

    def get_elbow_orientation(self, target_pose):

        dx, dy, dz = target_pose
        euc_dist = sqrt( dx**2 + dy**2 + dz**2 )
        if euc_dist > 36.00/100: # 36 cm
            return 1 # Should do elbow_down
        else:
            return 0 # Should do elbow_up

    def get_pitch_based_on_target_distance(self, target_pose):
        dx, dy, dz = target_pose
        euc_dist = sqrt( dx**2 + dy**2 + dz**2 )
        if euc_dist < 36.00/100: # 24 cm
            return 80*D2R  # Constant 90 deg Pitch
        else:
            return 80*D2R # Constant 0 deg Pitch


    ### @brief Enters Pick and Place state
    def pick_place(self):
        self.current_state = "pick_place"
        self.next_state = "execute"
        self.waypoints = []
        self.replay_buffer = []

        i = 0
        
        # collect and execute two waypoints
        while i < 2:
            # every time we get a new click, find joint angles and add to list of waypoints
            if self.camera.new_click == True:
                # get last click coordinates and transform it to world coordinates
                pt = self.camera.last_click
                z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
                pt_in_world = self.camera.transform_pixel_to_world(pt[0], pt[1])
                pt_in_world = np.append(pt_in_world , (972.0 - z)/1000  )

                # wrist angles for grasp in radians
                phi =  0
                theta = 0 
                # psi = 60 * D2R # Maintain a constant pitch 60
                # if < 20cm : set pitch(psi) to 90 deg
                # else: set pitch to 0 deg
                psi = self.get_pitch_based_on_target_distance(pt_in_world)
                
                pose = np.append(pt_in_world, np.array([phi, theta, psi]))
                
                # Append a pose just above it
                waypoint_offset = 0.06 # 4cm above
                inter_waypoint = np.array([pt_in_world[0], pt_in_world[1], pt_in_world[2]+waypoint_offset])
                inter_elbow_status = self.get_elbow_orientation(inter_waypoint) # 0 is up, 1 is down
                print("inter_elbow_status: (0:up, 1: down) ", inter_elbow_status)
                first_pose = np.append( inter_waypoint, np.array([phi, theta, psi]))
                
                print("Inter waypoint: ", first_pose)
                upper_joint_positions = IK_geometric(self.rxarm.dh_params, first_pose)
                upper_joint_positions = self.rxarm.find_best_soluton(upper_joint_positions, inter_elbow_status, pose[2]) # pose[2] is the target_z
                print("add first waypoint: (deg)", upper_joint_positions*R2D)
                self.waypoints.append(upper_joint_positions) # Takes elbow_down sol
                if i==0:
                    self.replay_buffer.append(1) # Open gripper
                else:
                    self.replay_buffer.append(0) # Current carrying a block, don't open gripper
                
                # Add Target Point
                elbow_status = self.get_elbow_orientation(pt_in_world) # 0 is up, 1 is down
                print("elbow_status: (0:up, 1: down) ", elbow_status)   
                if i==0:
                    pose[2] -= 0.02
                else:
                    pose[2] += 0.012 # add 2cm in z when "placing" the block
                joint_positions = IK_geometric(self.rxarm.dh_params, pose)
                joint_positions = self.rxarm.find_best_soluton(joint_positions, elbow_status, pose[2])

                if i==0:
                    self.replay_buffer.append(-1) # Close gripper
                else:                         
                    self.replay_buffer.append(1) # Now reach target, placing: Open gripper
                self.waypoints.append(joint_positions)
                print("add TARGET waypoint: (deg)", joint_positions*R2D)

                # Append End pose just above it
                print("add end waypoint: (deg)", upper_joint_positions*R2D)
                self.waypoints.append(upper_joint_positions) # Takes elbow_down sol
                self.replay_buffer.append(0) # Don't change gripper state

                # advance index and get ready for next click
                i = i + 1
                self.camera.new_click = False
        
            

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.rxarm.enable_torque()
        if len(self.waypoints) == 0:
            waypoints = [[-np.pi/2,       -0.5,      -0.3,      0.0,       0.0   ],
                    [0.75*-np.pi/2,   0.5,       0.3,      0.0,      np.pi/2],
                    [0.5*-np.pi/2,   -0.5,      -0.3,     np.pi/2,    0.0   ],
                    [0.25*-np.pi/2,   0.5,       0.3,      0.0,      np.pi/2],
                    [0.0,             0.0,       0.0,      0.0,       0.0   ],
                    [0.25*np.pi/2,   -0.5,      -0.3,      0.0,      np.pi/2],
                    [0.5*np.pi/2,     0.5,       0.3,     np.pi/2,    0.0   ],
                    [0.75*np.pi/2,   -0.5,      -0.3,      0.0,      np.pi/2],
                    [np.pi/2,         0.5,       0.3,      0.0,       0.0   ],
                    [0.0,             0.0,       0.0,      0.0,       0.0   ]]

            for _ in range(len(waypoints)):
                self.replay_buffer.append(0)

        else:
            waypoints = self.waypoints

        # send kinematics
        print('executing...')
        max_errors = {}
        # print("Robot waypoints: ", waypoints)

        # TODO : 11/16 Tue : add a 'FLAG' to specify whether we are now closing: 
        # 1: we are currently CLOSED. when we reach: we want to open
        # 0: we are currently OPEN. When we reach: we want to close the gripper
        state = 0 # Initially: OPEN
        index = 0
        for wp in waypoints:

            # Note: We should set a 'flag' here for execution.
            # If it is a 'Teach and Repeat' task: open and close gripper in between
            # open gripper
            # self.rxarm.open_gripper()
            # rospy.sleep(2)
            self.rxarm.set_positions(wp)
            rospy.sleep(2)
            if self.replay_buffer[index] == 0:
                # just go to this waypoint; nothing required
                pass
            elif self.replay_buffer[index] == 1: # open
                self.rxarm.open_gripper()
                rospy.sleep(1.5)
            else: # close gripper -1
                self.rxarm.close_gripper()
                rospy.sleep(1.5)

            print(wp)
            # sleep to get to waypoint\
            # rospy.sleep(3)
            # print("error: " , list(self.rxarm.get_joint_positions()) - wp )
            total_errors = []
            for i in range(len(wp)):
                diff = abs(self.rxarm.get_joint_positions()[i] - wp[i])
                total_errors.append(diff)
            max_errors[self.rxarm.joint_names[np.argmax(total_errors)]] = max(total_errors)
            # print('errors: ', total_errors)

            # increment index
            index += 1

        # print("max errors in each waypoint: ", max_errors)
        self.status_message = "State: Execute - Executing motion plan"
        self.next_state = "idle"
        self.waypoints = []
        # clear replay buffer
        self.replay_buffer = []
        self.rxarm.sleep()


    def execute_event3(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.rxarm.enable_torque()
        if len(self.waypoints) == 0:
            waypoints = [[-np.pi/2,       -0.5,      -0.3,      0.0,       0.0   ],
                    [0.75*-np.pi/2,   0.5,       0.3,      0.0,      np.pi/2],
                    [0.5*-np.pi/2,   -0.5,      -0.3,     np.pi/2,    0.0   ],
                    [0.25*-np.pi/2,   0.5,       0.3,      0.0,      np.pi/2],
                    [0.0,             0.0,       0.0,      0.0,       0.0   ],
                    [0.25*np.pi/2,   -0.5,      -0.3,      0.0,      np.pi/2],
                    [0.5*np.pi/2,     0.5,       0.3,     np.pi/2,    0.0   ],
                    [0.75*np.pi/2,   -0.5,      -0.3,      0.0,      np.pi/2],
                    [np.pi/2,         0.5,       0.3,      0.0,       0.0   ],
                    [0.0,             0.0,       0.0,      0.0,       0.0   ]]

            for _ in range(len(waypoints)):
                self.replay_buffer.append(0)

        else:
            waypoints = self.waypoints

        # send kinematics
        print('execute_event3...')
        max_errors = {}
        # print("Robot waypoints: ", waypoints)

        # TODO : 11/16 Tue : add a 'FLAG' to specify whether we are now closing: 
        # 1: we are currently CLOSED. when we reach: we want to open
        # 0: we are currently OPEN. When we reach: we want to close the gripper
        state = 0 # Initially: OPEN
        index = 0
        for wp in waypoints:

            # Note: We should set a 'flag' here for execution.
            # If it is a 'Teach and Repeat' task: open and close gripper in between
            # open gripper
            # self.rxarm.open_gripper()
            # rospy.sleep(2)
            self.rxarm.set_positions(wp)
            rospy.sleep(2)
            if self.replay_buffer[index] == 0:
                # just go to this waypoint; nothing required
                pass
            elif self.replay_buffer[index] == 1: # open
                self.rxarm.open_gripper()
                rospy.sleep(1)
            else: # close gripper -1
                self.rxarm.close_gripper()
                rospy.sleep(1)

            print(wp)
            # sleep to get to waypoint\
            # rospy.sleep(3)
            # print("error: " , list(self.rxarm.get_joint_positions()) - wp )
            total_errors = []
            for i in range(len(wp)):
                diff = abs(self.rxarm.get_joint_positions()[i] - wp[i])
                total_errors.append(diff)
            max_errors[self.rxarm.joint_names[np.argmax(total_errors)]] = max(total_errors)
            # print('errors: ', total_errors)

            # increment index
            index += 1

        # print("max errors in each waypoint: ", max_errors)
        # self.status_message = "State: Execute_event3 - Executing motion plan"
        # self.next_state = "line_em_up"
        self.waypoints = []
        # clear replay buffer
        self.replay_buffer = []
        self.rxarm.sleep()



    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        self.status_message = "Calibrating using Apriltag locations..."


        pt_world = self.camera.world_apriltag_coords
        pt_camera = np.zeros_like(pt_world)

        # print("Detect " , self.camera.tag_detectionss, " apriltags")

        # put apriltag locations (shown in camera frame) into an array
        for i in range(len(pt_world)) :
            x = self.camera.tag_detections.detections[i].pose.pose.pose.position.x
            y = self.camera.tag_detections.detections[i].pose.pose.pose.position.y
            z = self.camera.tag_detections.detections[i].pose.pose.pose.position.z                        
            id =  self.camera.tag_detections.detections[i].id
            pt_camera[id[0]-1] = np.array([x/z, y/z, 1])
            pt_world[i] = np.asarray(pt_world[i])

        # define parameters for solvePnP, format
        dist_coeffs = self.camera.dist_coeffs
        flag = 0
        pt_camera = np.asarray(pt_camera)

        # convert apriltag locations from camera frame to image frame
        pt_image_T = np.matmul(self.camera.intrinsic_matrix , pt_camera.T)


        # print("pt_world: " , pt_world)
        # print("pt_camera: " , pt_camera)
        # print("pt_image: ", pt_image_T.T)

        # delete last column of image frame points, format inputs for solvePnP
        pt_image = pt_image_T.T
        pt_image = np.delete(pt_image, 2, axis=1)
        pt_world.astype(np.float64)
        pt_image.astype(np.float64)

        # find rotation and translation vectors between camera frame and world frame
        success, Rot , trans = cv2.solvePnP(pt_world , pt_image , self.camera.intrinsic_matrix , dist_coeffs , flags=flag)

        #TEST
        #trans = trans * -1

        # convert rotation vector to homogeneous rotation matrix (3 by 3)
        Rot_3by3 , jacob = cv2.Rodrigues(Rot)

        # print("Rot: ", Rot)
        # print("Rot_3by3: ", Rot_3by3)
        # print("trans: ", trans)

        # assemble extrinsic matrix
        bot_row = np.array([0 , 0 , 0 , 1])
        self.camera.extrinsic_matrix = np.vstack((np.hstack((Rot_3by3 , trans)) , bot_row))

        # print("EXTRINSIC: " , self.camera.extrinsic_matrix)
        
        # Add 1 to each pt in world_apriltag_coords   
        ones = np.array([1, 1, 1, 1]).T.reshape(-1,1)
        pt_world = np.hstack(( pt_world, ones)) 

        # verify accuracy of extrinsic matrix
        pt_camera_verified = np.matmul(self.camera.extrinsic_matrix, pt_world.T)
        # print("Ours pt_camera: ", pt_camera_verified.T) # row by row
        # print("pt_camera from aprilTag: ", pt_camera)
        
        
        rot_part = self.camera.extrinsic_matrix[0:2, 0:2]
        trans_part = self.camera.extrinsic_matrix[0:2, 3]
        # print("rot_part: ", rot_part)
        # print("trans_part: ", trans_part)
        self.camera.extrinsic_matrix = np.vstack( ( np.hstack((rot_part, trans_part.T.reshape(2,1))) , np.array([0, 0, 1]).reshape(1,3) ) )
        # self.camera.extrinsic_matrix[:,0] = -self.camera.extrinsic_matrix[:,0]
        print("extrinsic matrix 3x3: ", self.camera.extrinsic_matrix)


        self.cameraCalibrated = True
        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"


    ########### Competition #############

    def pick_n_sort(self):
        """
        @brief sorts blocks in positive half plane. large blocks go to the right negative
            half plane and small blocks go to the left negative half plane. blocks are
            sorted in rainbow order
        """

        
        self.current_state = "pick_n_sort"
        self.next_state = "execute"

        self.status_message = "Event1 : Pick n sort"



        # Level1: R,G,B Large block
        """
            1. camera.block_detectors: get a list of blocks
            2. use a container to store blocks info
            3. while there's still block in the container: do
               - set (block.x,y,z, phi, theta=ori, psi) to IK_geometric
               - add (firts, target, end) waypoints to execute
               - select a location in the negative half plane ( 12 pre-defined locations)
               - add (first, destination, end) waypoints
        """
        
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        # Get current block info from block detector, add relevant offsets
        Blocks = self.camera.blockDetector()
        x_offset = 0.1 # 12cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        # large destinations
        dest_poses = [
            [0.2 + 2*x_offset, -0.03, 0, phi, theta, psi],
            [0.2 +   x_offset, -0.03, 0, phi, theta, psi],
            [0.2             , -0.03, 0, phi, theta, psi],
            [0.2 + 2*x_offset, -0.12, 0, phi, theta, psi],
            [0.2 +   x_offset, -0.12, 0, phi, theta, psi],
            [0.2             , -0.12, 0, phi, theta, psi]
        ]

        # small destinations
        z_sm = +0.005 # - 1 cm
        dest_poses_sm = [
            [-0.2 - 2*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.2 -   x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.2             , -0.05,  z_sm, phi, theta, psi],
            [-0.2 - 2*x_offset, -0.12,  z_sm, phi, theta, psi],
            [-0.2 -   x_offset, -0.12,  z_sm, phi, theta, psi],
            [-0.2             , -0.12,  z_sm, phi, theta, psi]
        ]


        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)

        # Sort Blocks in R,G,B order
        Blocks.sort(key=my_custom_sort)

        i = 0
        index_lg = 0 # dest index for large blocks
        index_sm = 0
        
        # Blocks is NOT empty
        while Blocks:

            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.07 # 7cm

            # If the block is close to the robot, add more z offset
            # if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
            #     block.wz = 0.00
            # elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
            #     block.wz =0.01 #
            # else:
            #     block.wz =0.032 
            
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.005 # -2cm

            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy)) + np.pi/2
            print("block ori: ",degrees(block.ori))
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            highwp = deepcopy(first_wp)
            highwp[3] = -highwp[3]
            self.waypoints.append(highwp)
            self.replay_buffer.append(0) # hold

            highwp2 = deepcopy(highwp)
            highwp2[1] = radians(0)
            highwp2[2] = radians(90)
            highwp2[3] =  radians(0)
            self.waypoints.append(highwp2)
            self.replay_buffer.append(0) # hold


            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            # dest_z_offset = -0.008 # -0.8 cm
            if sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 18.00/100:
                dest_z_offset = -0.02 # -0.8 cm
            else:
                z_offset = 0.08       # interwaypoint offset: 8cm
                dest_z_offset = +0.02 # +1.0 cm

            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset
            # dest_joint wrist rotation:
            dest_pose[4] = 0 + (np.pi/2 - atan2(dest_pose[0], dest_pose[1]))


            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold
            
            highwp = deepcopy(df_wp)
            highwp[3] = -highwp[3]
            self.waypoints.append(highwp)
            self.replay_buffer.append(0) # hold

            highwp2 = deepcopy(highwp)
            highwp2[1] = radians(0)
            highwp2[2] = radians(90)
            highwp2[3] = radians(0)
            self.waypoints.append(highwp2)
            self.replay_buffer.append(0) # hold


            # Move to the next block:
            i = i+1
            # for wp in self.waypoints:
            #     print(wp)  
                      

        for j in range(len(self.waypoints)):
            print(self.waypoints[j])
            if self.replay_buffer[j]==0:
                print("hold")
            elif self.replay_buffer[j]==1:
                print("Open")
            else:
                print("Close")
             
        print("Done event1")

    def pick_n_sort_wrong(self):


        self.current_state = "pick_n_sort"
        self.status_message = "Event1 : Pick n sort"



        # Level1: R,G,B Large block
        """
            1. camera.block_detectors: get a list of blocks
            2. use a container to store blocks info
            3. while there's still block in the container: do
               - set (block.x,y,z, phi, theta=ori, psi) to IK_geometric
               - add (firts, target, end) waypoints to execute
               - select a location in the negative half plane ( 3 pre-defined locations)
               - add (first, destination, end) waypoints
        """
        
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []



        # First : Move ALL negative blocks to positive plane
        # block.wy <= 0.05 : move to 0.1

        homepose = np.array([0, 0, 0, 0, 0])
        # self.waypoints.append(homepose)
        # self.replay_buffer.append(1)
        self.rxarm.initialize()
        # self.rxarm.set_positions(list(homepose))
        # rospy.sleep(2)

        phi = 0.0
        psi = radians(80)


        Blocks = self.camera.blockDetector()
        BlocksNeg = []
        for block in Blocks:
            if block.wy <= 0.05:
                BlocksNeg.append(block)

        print(str(len(BlocksNeg))+" neg block. Move them to positive plane! (y=10cm)")
        while BlocksNeg:
            print("Moving block ", block.color)
            # pop first and move it to block.wx, 0.1, block.wz
            block = BlocksNeg.pop(0)

            if sqrt(block.wx**2 + block.wy**2) <= 15.00/100:
                z_offset -= 0.07
            elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
                z_offset = -0.05 # -3cm
            else:
                z_offset = -0.02 # 1cm
            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
            block_position = np.array([block.wx, block.wy, block.wz + z_offset])
            block_elbow_status = self.get_elbow_orientation(block_position)
            block_pose = np.append(block_position, np.array([phi, theta, psi]))

            up_pose= np.array([block.wx, block.wy, block.wz+ z_offset +0.06, phi, theta, psi])
            up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)

            wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
            drop_pose = np.array([block.wx, 0.1, block.wz+ z_offset, phi, theta, psi])
            drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, block.wz)

            up_drop_pose= np.array([block.wx, 0.1, block.wz+ z_offset+0.06, phi, theta, psi])
            up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, block.wz)
    
            ########### Execute wps & gripper here ########            
            self.rxarm.open_gripper()
            rospy.sleep(1.5)
            self.rxarm.set_positions(up_wp)
            rospy.sleep(2)
            self.rxarm.set_positions(wp)
            rospy.sleep(2)
            self.rxarm.close_gripper()
            rospy.sleep(1.5)
            self.rxarm.set_positions(up_wp)
            rospy.sleep(2)

            self.rxarm.set_positions(up_drop_wp)
            rospy.sleep(2)
            self.rxarm.set_positions(drop_wp)
            rospy.sleep(2)
            self.rxarm.open_gripper()
            rospy.sleep(1.5)
            self.rxarm.set_positions(up_drop_wp)
            rospy.sleep(2)
            ###############################################
        print("Sleeping arm and take pictures..... ")
        self.rxarm.sleep()
        rospy.sleep(2.5)
        print("Detecting blocks....")
        # while True:
        #     pass
            
        ### 1208 TODO : how to efficiently unstack? ######
        ########## Then, Unstack if needed ################

        """
            Unstack Logic :
            1. Use a for-loop for 4 levels
            2. Scan, if no block detected: go to lower level
            3. If block detected: move that block to its "left"
            4. repeat until we reach 1st level (stop)

            Expected z high:
                15 + 
                11 +
                7 +

            Note: use blockDetector_givenheight
        """

        # heights = [15.00, 11.00, 7.00]
        heights = [8.00, 5.00, 2.00]
        dirs = ["left", "down", "right"]

        bh = 30 # block height: 30,
        centers = [830,  866 , 904]
        widths = [20, 8, 3]

        for i in range(3): ## Do 3 times (from level 4 to level 2 )
            
            Blocks = self.camera.blockDetector_givenheight(centers[i], widths[i])
            print("Unstacking level " +str(4-i))
            print("Detect "+str(len(Blocks))+ " blocks")

            while Blocks:
                block = Blocks.pop(0)
                if block.wz <= heights[i]/100 : # convert to meter
                    print("Too low, not this level so don't move.")
                else:
                    print("unstacking block "+block.color)
                    # Move this block to the dirs[i]
                    ######## For z offset ####
                    if sqrt(block.wx**2 + block.wy**2) <= 19.00/100:
                        z_offset = -0.07
                    elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
                        z_offset = -0.04 # -3cm
                    else:
                        z_offset = -0.02 # 1cm

                    theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
                    
                    # if abs(block.ori - np.pi/2) <= radians(10):
                        # block.wx -= 0.01 # x-1cm

                    block_position = np.array([block.wx, block.wy, block.wz + z_offset])
                    block_elbow_status = self.get_elbow_orientation(block_position)
                    block_pose = np.append(block_position, np.array([phi, theta, psi]))
                    up_pose= np.array([block.wx, block.wy, block.wz+ z_offset +0.06, phi, theta, psi])
                    up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)
                    wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
                    
                    dx = 0.125 # 12.5 cm
                    dy = 0.125
                    z_offset = +0.02 #

                    if dirs[i] == "left":
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx - dx, block.wy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx - dx, block.wy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)
                        
                    elif dirs[i]=="down":
                        drop_pose = np.array([block.wx, block.wy - dy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx, block.wy - dy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                        
                    else: # right
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx + dx, block.wy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx + dx, block.wy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                      

                    ########### Execute wps & gripper here ########            
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(wp)
                    rospy.sleep(2)
                    self.rxarm.close_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)

                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(drop_wp)
                    rospy.sleep(2)
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    ###############################################
            print("Sleep arm and check lower level:")
            self.rxarm.sleep()
            rospy.sleep(2)
        # return
        print("Done unsatcking")
        ###################################################

        print("Now placing them...")

        Blocks = self.camera.blockDetector()
        x_offset = 0.05 # 5cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        dest_poses = [
            [0.2 + 4*x_offset, -0.008, 0, phi, theta, psi],
            [0.2 + 3*x_offset, -0.008, 0, phi, theta, psi],
            [0.2 + 2*x_offset, -0.008, 0, phi, theta, psi],
            [0.2 + 1*x_offset, -0.008, 0, phi, theta, psi],
            [0.2 + 0*x_offset, -0.008, 0, phi, theta, psi],
            [0.2 + 4*x_offset, -0.015, 0, phi, theta, psi],
            [0.2 + 3*x_offset, -0.015, 0, phi, theta, psi],
            [0.2 + 2*x_offset, -0.015, 0, phi, theta, psi],
            [0.2 + 1*x_offset, -0.015, 0, phi, theta, psi]
        ]

        # small destinations
        z_sm = -0.01 # - 1 cm
        dest_poses_sm = [
            [-0.2 - 4*x_offset, -0.008, 0, phi, theta, psi],
            [-0.2 - 3*x_offset, -0.008, 0, phi, theta, psi],
            [-0.2 - 2*x_offset, -0.008, 0, phi, theta, psi],
            [-0.2 - 1*x_offset, -0.008, 0, phi, theta, psi],
            [-0.2 - 0*x_offset, -0.008, 0, phi, theta, psi],
            [-0.2 - 4*x_offset, -0.015, 0, phi, theta, psi],
            [-0.2 - 3*x_offset, -0.015, 0, phi, theta, psi],
            [-0.2 - 2*x_offset, -0.015, 0, phi, theta, psi],
            [-0.2 - 1*x_offset, -0.015, 0, phi, theta, psi]
        ]

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)




        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)


        index_lg = 0 # dest index for large blocks
        index_sm = 0
        Blocks = self.camera.blockDetector()
        Blocks.sort(key=my_custom_sort)
        while Blocks:
            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.07 # 7cm
            if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
                block.wz = 0.00 # 2cm
            elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
                block.wz = 0.01 # 2cm
            else:
                block.wz = 0.032 # here
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm

            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))    
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            # dest_z_offset = -0.008 # -0.8 cm
            if sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 18.00/100:
                dest_z_offset = -0.045 # -0.8 cm
            elif sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 23.00/100:
                dest_z_offset = -0.02 # -0.8 cm
            else:
                z_offset = 0.08       # interwaypoint offset: 8cm
                dest_z_offset = +0.02 # +1.0 cm


            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset

            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold


        print("Done event 1")
        self.next_state = "execute"

    def pick_n_sort_stacked(self):


        self.current_state = "pick_n_sort"
        self.status_message = "Event1 : Pick n sort"



        # Level1: R,G,B Large block
        """
            1. camera.block_detectors: get a list of blocks
            2. use a container to store blocks info
            3. while there's still block in the container: do
               - set (block.x,y,z, phi, theta=ori, psi) to IK_geometric
               - add (firts, target, end) waypoints to execute
               - select a location in the negative half plane ( 3 pre-defined locations)
               - add (first, destination, end) waypoints
        """
        
        self.rxarm.enable_torque()
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        Blocks = self.camera.blockDetector()
        x_offset = 0.1 # 12cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        dest_poses = [
            [0.2 + 2*x_offset, -0.05, 0, phi, theta, psi],
            [0.2 + x_offset,   -0.05, 0, phi, theta, psi],
            [0.2           ,   -0.05, 0,  phi, theta, psi],
            [0.2 + 2*x_offset, -0.09, 0, phi, theta, psi],
            [0.2 + x_offset, -0.09, 0, phi, theta, psi],
            [0.2, -0.09, 0 ,  phi, theta, psi]
        ]

        # small destinations
        z_sm = -0.01 # - 1 cm
        dest_poses_sm = [
            [-0.2 - 2*x_offset, -0.005,  z_sm, phi, theta, psi],
            [-0.2 - x_offset, -0.005, z_sm, phi, theta, psi],
            [-0.2, -0.005,  z_sm ,  phi, theta, psi],
            [-0.2 - 2*x_offset, -0.075, z_sm, phi, theta, psi],
            [-0.2 - x_offset, -0.075,  z_sm, phi, theta, psi],
            [-0.2, -0.075,  z_sm,  phi, theta, psi]
        ]


        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)

        # Sort Blocks in R,G,B order
        # Blocks.sort(key=my_custom_sort)
        Blocks.sort(key=lambda x: x.wz, reverse=True)

        index_lg = 0 # dest index for large blocks
        index_sm = 0
        # Blocks is NOT empty
        while Blocks:

            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.07 # 7cm

            if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
                block.wz -= 0.025 # -5cm
            elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
                block.wz -= 0.025 #  down 2.5cm more
            else:
                block.wz -= 0.025 # =0.032 
            
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm
            
            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            # dest_z_offset = -0.008 # -0.8 cm
            if sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 18.00/100:
                dest_z_offset = -0.02 # -0.8 cm
            else:
                z_offset = 0.08       # interwaypoint offset: 8cm
                dest_z_offset = +0.02 # +1.0 cm

            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset


            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold
            


            ########### Execute wps & gripper here ########            
            index = 0
            for wp in self.waypoints:
                self.rxarm.set_positions(wp)
                rospy.sleep(2)
                if self.replay_buffer[index] == 0:
                    # just go to this waypoint; nothing required
                    pass
                elif self.replay_buffer[index] == 1: # open
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                else: # close gripper -1
                    self.rxarm.close_gripper()
                    rospy.sleep(1.5)

                # increment index
                index += 1
            ###############################################


            # Clear all queue
            self.waypoints = [homepose]
            self.replay_buffer = [1] # Open

            self.rxarm.sleep()
            rospy.sleep(4)
            Blocks = self.camera.blockDetector()
            Blocks.sort(key=lambda x: x.wz, reverse=True)


            BlocksLeft = []
            for j in range(len(Blocks)):
                print(Blocks[j].color, " y: ", Blocks[j].wy)
                if Blocks[j].wy <= 0.0:
                    print("Remove", Blocks[j].color, "y : ", Blocks[j].wy)
                else:
                    BlocksLeft.append(Blocks[j])
            Blocks = BlocksLeft

            print("After removing: number of blocks: ", len(Blocks))
            print(Blocks)   
            print("\n\n")                   


        for j in range(len(self.waypoints)):
            print(self.waypoints[j])
            if self.replay_buffer[j]==0:
                print("hold")
            elif self.replay_buffer[j]==1:
                print("Open")
            else:
                print("Close")
             
        print("Done event1")

    def pick_n_stack(self):

        self.current_state = "pick_n_stack"
        self.status_message = "Event2 : Pick n stack"
        # Level1: R,G,B Large block
        """
            1. camera.block_detectors: get a list of blocks
            2. use a container to store blocks info
            3. while there's still block in the container: do
               - set (block.x,y,z, phi, theta=ori, psi) to IK_geometric
               - add (firts, target, end) waypoints to execute
               - select a location in the negative half plane ( 3 pre-defined locations)
               - add (first, destination, end) waypoints
        """
        
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        Blocks = self.camera.blockDetector()
        # x_offset = 0.1 # 12cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        # block_height = 0.0375
        # block_height_sm = 0.024 # 2.4cm
        block_height0 = 0.05
        block_height1 = 0.035
        block_height2 = 0.0275
        x_offset0 = 0.01
        x_offset1 = 0.008
        x_offset2 = 0.0075 # 0.5cm
        z0 = 0.01 # 0.8 cm
        z1 = -0.005
        z2 = -0.03
        dest_poses = [
            [0.35, -0.03,             z0, phi, theta, psi],
            [0.35 +  x_offset0 -0.004, -0.03, z0 + block_height0, phi, theta, psi],
            [0.35 + 2*x_offset0, -0.03, z0 + 2*block_height0-0.008,  phi, theta, psi],
            [0.3, -0.03, z1, phi, theta, psi],
            [0.3+x_offset1, -0.03, z1 + block_height1, phi, theta, psi],
            [0.3+2*x_offset1, -0.03, z1 +2*block_height1+0.002,  phi, theta, psi],
            [0.18, -0.03, z2, phi, theta, psi],
            [0.18+x_offset2, -0.03, z2 + block_height2, phi, theta, psi],
            [0.18+2*x_offset2, -0.03, z2 +2*block_height2+0.002,  phi, theta, psi]
        ]
        
        block_height_sm0 = 0.025 # 2.4cm        
        block_height_sm1 = 0.018 # 2.4cm
        block_height_sm2 = 0.0185 # 2.4cm


        ######### TODO: tuning small stack params
        # small destinations
        z_sm0 =  0.01 # 1cm
        z_sm1 =  -0.005 
        z_sm2 =  0.01 # - 1 cm
        x_offset_sm0 = 0.002 #0.005
        x_offset_sm1 = 0.004 #0.0018
        x_offset_sm2 = 0.002 # 0.2cm

        # level3:
        dest_poses_sm = [
            [-0.4                       , -0.03,   z_sm0+0.01                       , phi, theta, psi],
            [-0.4-x_offset_sm0           , -0.03,   z_sm0+ block_height_sm0 +0.02, phi, theta, psi],
            [-0.4-2*x_offset_sm0-0.004  , -0.03,   z_sm0+2*block_height_sm0 +0.02,  phi, theta, psi],
            [-0.3                       , -0.03,   z_sm1, phi, theta, psi],
            [-0.3-x_offset_sm1          , -0.03,   z_sm1 + block_height_sm1, phi, theta, psi],
            [-0.3-2*x_offset_sm1        , -0.03,   z_sm1+2*block_height_sm1,  phi, theta, psi],
            [-0.22                      , -0.03, z_sm2, phi, theta, psi],
            [-0.22-x_offset_sm2         , -0.03, z_sm2 + block_height_sm2, phi, theta, psi],
            [-0.22-2*x_offset_sm2       , -0.03, z_sm2+2*block_height_sm2,  phi, theta, psi]
        ]

        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)

        # Sort Blocks in R,G,B order
        Blocks.sort(key=my_custom_sort)

        i = 0
        index_lg = 0 # dest index for large blocks
        index_sm = 0
        # Blocks is NOT empty
        while Blocks:

            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.08 # 7cm
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.000 # -2cm
                psi = radians(85)               


            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy)) # + np.pi/2
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            if sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 18.00/100:
                dest_z_offset = -0.02 # -0.8 cm
            else:
                z_offset = 0.08       # interwaypoint offset: 8cm
                dest_z_offset = +0.02 # +1.0 cm

            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset
            # dest_joint wrist rotation:
            dest_pose[4] = 0 + (np.pi/2 - atan2(dest_pose[0], dest_pose[1]))


            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold
            
            # Move to the next block:
            i = i+1  
                    
        print("Done event2")
        self.next_state = "execute"


    def line_em_up(self):


        self.current_state = "line_em_up"
        # self.next_state = "execute"
        self.status_message = "Event3: Line em up"
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []


        # First : Move ALL negative blocks to positive plane
        # block.wy <= 0.05 : move to 0.1

        homepose = np.array([0, 0, 0, 0, 0])
        # self.waypoints.append(homepose)
        # self.replay_buffer.append(1)
        self.rxarm.initialize()
        self.rxarm.set_positions(list(homepose))
        rospy.sleep(2)

        phi = 0.0
        psi = radians(80)


        # Blocks = self.camera.blockDetector()
        # BlocksNeg = []
        # for block in Blocks:
        #     if block.wy <= 0.05:
        #         BlocksNeg.append(block)

        # print(str(len(BlocksNeg))+" neg block. Move them to positive plane! (y=10cm)")
        # while BlocksNeg:
        #     print("Moving block ", block.color)
        #     # pop first and move it to block.wx, 0.1, block.wz
        #     block = BlocksNeg.pop(0)

        #     if sqrt(block.wx**2 + block.wy**2) <= 15.00/100:
        #         z_offset -= 0.07
        #     elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
        #         z_offset = -0.05 # -3cm
        #     else:
        #         z_offset = -0.02 # 1cm
        #     theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
        #     block_position = np.array([block.wx, block.wy, block.wz + z_offset])
        #     block_elbow_status = self.get_elbow_orientation(block_position)
        #     block_pose = np.append(block_position, np.array([phi, theta, psi]))

        #     up_pose= np.array([block.wx, block.wy, block.wz+ z_offset +0.06, phi, theta, psi])
        #     up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)

        #     wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
        #     drop_pose = np.array([block.wx, 0.1, block.wz+ z_offset, phi, theta, psi])
        #     drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, block.wz)

        #     up_drop_pose= np.array([block.wx, 0.1, block.wz+ z_offset+0.06, phi, theta, psi])
        #     up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, block.wz)
    
        #     ########### Execute wps & gripper here ########            
        #     self.rxarm.open_gripper()
        #     rospy.sleep(1.5)
        #     self.rxarm.set_positions(up_wp)
        #     rospy.sleep(2)
        #     self.rxarm.set_positions(wp)
        #     rospy.sleep(2)
        #     self.rxarm.close_gripper()
        #     rospy.sleep(1.5)
        #     self.rxarm.set_positions(up_wp)
        #     rospy.sleep(2)

        #     self.rxarm.set_positions(up_drop_wp)
        #     rospy.sleep(2)
        #     self.rxarm.set_positions(drop_wp)
        #     rospy.sleep(2)
        #     self.rxarm.open_gripper()
        #     rospy.sleep(1.5)
        #     self.rxarm.set_positions(up_drop_wp)
        #     rospy.sleep(2)
        #     ###############################################
        # print("Sleeping arm and take pictures..... ")
        # self.rxarm.sleep()
        # rospy.sleep(2.5)
        print("Detecting blocks....")
        # while True:
        #     pass
            
        ### 1208 TODO : how to efficiently unstack? ######
        ########## Then, Unstack if needed ################

        """
            Unstack Logic :
            1. Use a for-loop for 4 levels
            2. Scan, if no block detected: go to lower level
            3. If block detected: move that block to its "left"
            4. repeat until we reach 1st level (stop)

            Expected z high:
                15 + 
                11 +
                7 +

            Note: use blockDetector_givenheight
        """

        # heights = [15.00, 11.00, 7.00]
        heights = [8.00, 5.00, 2.00]
        dirs = ["left", "down", "left"]

        bh = 30 # block height: 30,
        centers = [830,  866 , 904]
        widths = [20, 8, 8]

        for i in range(3): ## Do 3 times (from level 4 to level 2 )
            Blocks = self.camera.blockDetector_givenheight(centers[i], widths[i])
            print("Unstacking level " +str(4-i))
            print("Detect "+str(len(Blocks))+ " blocks")
            if not Blocks:
                print("Empty block. Jump to lower level")
                continue
            while Blocks:
                block = Blocks.pop(0)
                if block.wz <= heights[i]/100 : # convert to meter
                    print("Too low, not this level so don't move.")
                else:
                    print("unstacking block "+block.color)
                    # Move this block to the dirs[i]
                    ######## For z offset ####
                    if sqrt(block.wx**2 + block.wy**2) <= 19.00/100:
                        z_offset = -0.03
                    elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
                        z_offset = -0.015 # -3cm
                    else:
                        z_offset = -0.01 # 1cm

                    theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
                    
                    # if abs(block.ori - np.pi/2) <= radians(10):
                        # block.wx -= 0.01 # x-1cm

                    block_position = np.array([block.wx, block.wy, block.wz + z_offset])
                    block_elbow_status = self.get_elbow_orientation(block_position)
                    block_pose = np.append(block_position, np.array([phi, theta, psi]))
                    up_pose= np.array([block.wx, block.wy, block.wz+ z_offset+0.1, phi, theta, psi])
                    up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)
                    wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
                    
                    dx = 0.1 # 10 cm
                    dy = 0.1
                    z_offset = +0.02 #

                    if dirs[i] == "left":
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx - dx, block.wy-dy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx - dx, block.wy-dy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)
                        
                    elif dirs[i]=="down":
                        drop_pose = np.array([block.wx, block.wy - dy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx, block.wy - dy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                        
                    else: # right
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx + dx, block.wy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx + dx, block.wy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                      

                    ########### Execute wps & gripper here ########            
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(wp)
                    rospy.sleep(2)
                    self.rxarm.close_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)

                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(drop_wp)
                    rospy.sleep(2)
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    ###############################################
            print("Sleep arm and check lower level:")
            self.rxarm.sleep()
            rospy.sleep(2)
        # return
        print("Done unsatcking")
        ###################################################

        print("Now lining up...")

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        ########## Then, Line em up to the neg have plane ######
        x_offset = 0.06 # 5 cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        # dest_poses = [
        #     [0.4  - 5*x_offset       , -0.1, 0.065, phi, theta, psi],
        #     [0.4  - 4*x_offset+0.005 , -0.1, 0.03, phi, theta, psi],
        #     [0.4  - 3*x_offset       , -0.1, 0.01 , phi, theta, psi],
        #     [0.4  - 2*x_offset       , -0.1, 0   , phi, theta, psi],
        #     [0.4  - 1*x_offset       , -0.1, 0.01, phi, theta, psi],
        #     [0.4  - 0*x_offset       , -0.1, 0.01, phi, theta, psi]
        #     # [0.15             , -0.1, 0, phi, theta, psi]
        # ]

        dest_poses = [
            [0.4  - 0*x_offset       , -0.1, 0.01, phi, theta, psi],
            [0.4  - 1*x_offset       , -0.1, 0.01, phi, theta, psi],
            [0.4  - 2*x_offset       , -0.1, 0   , phi, theta, psi],
            [0.4  - 3*x_offset       , -0.1, 0   , phi, theta, psi],
            [0.4  - 4*x_offset       , -0.13, 0.03, phi, theta, psi],
            [0.4  - 5*x_offset       , -0.15, 0.03, phi, theta, psi]
            # [0.15             , -0.1, 0, phi, theta, psi]
        ]
        # small destinations
        z_sm = -0.01 # - 1 cm
        dest_poses_sm = [
            [-0.2 - 5*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.2 - 4*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.2 - 3*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.2 - 2*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.2 - x_offset  , -0.05,  z_sm, phi, theta, psi],
            [-0.2             , -0.05,  z_sm, phi, theta, psi]
        ]


        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)


        index_lg = 0 # dest index for large blocks
        index_sm = 0
        Blocks = self.camera.blockDetector()
        Blocks.sort(key=my_custom_sort)
        while Blocks:
            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.07 # 7cm
            # if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
            #     block.wz = 0.02 # 2cm
            # elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
            #     block.wz = 0.01 # 2cm
            # else:
            #     block.wz = 0.032 # here
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm

            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))    
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            # dest_z_offset = -0.008 # -0.8 cm
            if sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 18.00/100:
                dest_z_offset = -0.02 # -0.8 cm
            elif sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 23.00/100:
                dest_z_offset = -0.01 # -0.8 cm
            else:
                z_offset = 0.08       # interwaypoint offset: 8cm
                dest_z_offset = +0.02 # +1.0 cm


            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset

            # dest_joint wrist rotation:
            dest_pose[4] = 0 + (np.pi/2 - atan2(dest_pose[0], dest_pose[1]))

            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold


        print("Done event 3")
        self.next_state = "execute"

    def stack_em_high(self):


        self.current_state = "stack_em_up"
        # self.next_state = "execute"
        self.status_message = "Event4: Stack em up"
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []


        # First : Move ALL negative blocks to positive plane
        # block.wy <= 0.05 : move to 0.1

        homepose = np.array([0, 0, 0, 0, 0])
        # self.waypoints.append(homepose)
        # self.replay_buffer.append(1)
        self.rxarm.initialize()
        # self.rxarm.set_positions(list(homepose))
        # rospy.sleep(2)

        phi = 0.0
        psi = radians(80)

        ######## Clear right neg half plane (+x dir)
        # Blocks = self.camera.blockDetector()
        # BlocksNeg = []
        # for block in Blocks:
        #     if block.wy <= 0.05:
        #         BlocksNeg.append(block)

        # print(str(len(BlocksNeg))+" neg block. Move them to positive plane! (y=10cm)")
        # while BlocksNeg:
        #     print("Moving block ", block.color)
        #     # pop first and move it to block.wx, 0.1, block.wz
        #     block = BlocksNeg.pop(0)

        #     if sqrt(block.wx**2 + block.wy**2) <= 15.00/100:
        #         z_offset -= 0.07
        #     elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
        #         z_offset = -0.05 # -3cm
        #     else:
        #         z_offset = -0.02 # 1cm
        #     theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
        #     block_position = np.array([block.wx, block.wy, block.wz + z_offset])
        #     block_elbow_status = self.get_elbow_orientation(block_position)
        #     block_pose = np.append(block_position, np.array([phi, theta, psi]))

        #     up_pose= np.array([block.wx, block.wy, block.wz+ z_offset +0.06, phi, theta, psi])
        #     up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)

        #     wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
        #     drop_pose = np.array([block.wx, 0.1, block.wz+ z_offset, phi, theta, psi])
        #     drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, block.wz)

        #     up_drop_pose= np.array([block.wx, 0.1, block.wz+ z_offset+0.06, phi, theta, psi])
        #     up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, block.wz)
    
        #     ########### Execute wps & gripper here ########            
        #     self.rxarm.open_gripper()
        #     rospy.sleep(1.5)
        #     self.rxarm.set_positions(up_wp)
        #     rospy.sleep(2)
        #     self.rxarm.set_positions(wp)
        #     rospy.sleep(2)
        #     self.rxarm.close_gripper()
        #     rospy.sleep(1.5)
        #     self.rxarm.set_positions(up_wp)
        #     rospy.sleep(2)

        #     self.rxarm.set_positions(up_drop_wp)
        #     rospy.sleep(2)
        #     self.rxarm.set_positions(drop_wp)
        #     rospy.sleep(2)
        #     self.rxarm.open_gripper()
        #     rospy.sleep(1.5)
        #     self.rxarm.set_positions(up_drop_wp)
        #     rospy.sleep(2)
        #     ###############################################
        # print("Sleeping arm and take pictures..... ")
        # self.rxarm.sleep()
        # rospy.sleep(2.5)



        print("Detecting blocks....")
        # while True:
        #     pass
            
        ### 1208 TODO : how to efficiently unstack? ######
        ########## Then, Unstack if needed ################

        """
            Unstack Logic :
            1. Use a for-loop for 4 levels
            2. Scan, if no block detected: go to lower level
            3. If block detected: move that block to its "left"
            4. repeat until we reach 1st level (stop)

            Expected z high:
                15 + 
                11 +
                7 +

            Note: use blockDetector_givenheight
        """

        # heights = [15.00, 11.00, 7.00]
        heights = [8.00, 5.00, 2.00]
        dirs = ["right", "down", "left"]

        bh = 30 # block height: 30,
        centers = [830,  866 , 904]
        widths = [20, 8, 8]


        ## Relocate green and purple blocks!
        self.rxarm.go_to_home_pose()
        rospy.sleep(1)
        self.rxarm.open_gripper()
        rospy.sleep(1.5)
        self.rxarm.set_positions(green_up_config)
        rospy.sleep(2) 
        self.rxarm.set_positions(green_far_config)
        rospy.sleep(2)        
        self.rxarm.close_gripper()
        rospy.sleep(1.5)
        self.rxarm.set_positions(green_up_config)
        rospy.sleep(2) 
        self.rxarm.set_positions(drop_green_config)
        rospy.sleep(2)
        self.rxarm.open_gripper()
        rospy.sleep(1.5)


        # self.rxarm.set_positions(purple_up_config)
        # rospy.sleep(2) 
        # self.rxarm.set_positions(purple_config)
        # rospy.sleep(2)
        # self.rxarm.close_gripper()
        # rospy.sleep(1.5)
        # self.rxarm.set_positions(purple_up_config)
        # rospy.sleep(2) 
        # self.rxarm.set_positions(drop_purple_config)
        # rospy.sleep(2)
        # self.rxarm.open_gripper()
        # rospy.sleep(1.5)
        self.rxarm.go_to_home_pose()
        rospy.sleep(1.5)


        

        for i in range(3): ## Do 3 times (from level 4 to level 2 )
            
            Blocks = self.camera.blockDetector_givenheight(centers[i], widths[i])
            print("Unstacking level " +str(4-i))
            print("Detect "+str(len(Blocks))+ " blocks")
            if not Blocks:
                print("Empty block. Jump to lower level")
                continue
            while Blocks:
                block = Blocks.pop(0)
                if block.wz <= heights[i]/100 : # convert to meter
                    print("Too low, not this level so don't move.")
                else:
                    print("unstacking block "+block.color)
                    # Move this block to the dirs[i]
                    ######## For z offset ####
                    if sqrt(block.wx**2 + block.wy**2) <= 19.00/100:
                        z_offset = -0.06
                    elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
                        z_offset = -0.04 # -3cm
                    else:
                        z_offset = +0.03 # 1cm

                    theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
                    if block.wy >= 35.00/100:
                        psi = radians(10)
                        print("set pitch to 30 deg")
                    else:
                        psi = radians(80)
                    
                    if block.wz >= 0.03:
                        block.wz -= 0.03
                    # block.wx -= 0.005 # FOR THE ORANGE BLOCK!!!
                    # block.wy -= 0.005

                    block_position = np.array([block.wx, block.wy, block.wz + z_offset])
                    block_elbow_status = self.get_elbow_orientation(block_position)
                    block_pose = np.append(block_position, np.array([phi, theta, psi]))
                    up_pose= np.array([block.wx-0.1, block.wy-0.1, block.wz+ z_offset+0.1, phi, theta, psi])
                    up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)
                    wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
                    
                    dx = 0.09
                    dy = 0.09
                    z_offset = +0.04 #

                    if dirs[i] == "left":
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx - dx, block.wy-dy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx - dx, block.wy-dy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)
                        
                    elif dirs[i]=="down":
                        drop_pose = np.array([block.wx, block.wy - dy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx, block.wy - dy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                        
                    else: # right
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx + dx, block.wy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx + dx, block.wy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                      

                    ########### Execute wps & gripper here ########            
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(wp)
                    rospy.sleep(2)
                    self.rxarm.close_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)

                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(drop_wp)
                    rospy.sleep(2)
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    ###############################################
            self.rxarm.sleep()
            rospy.sleep(2)
        # return
        print("Done unsatcking")
        ###################################################

        print("Now stacking them high ...")

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        ########## Then, Line em up to the neg have plane ######
        x_offset = 0.015 # 6mm
        phi = 0.0
        theta=0.0
        block_h = 0.043 # 4cm
        psi = radians(80)
        dest_poses = [
            [0.15 + 0*x_offset         , -0.1, 0+ 0*block_h, phi, theta, psi],
            [0.15 + 1*x_offset         , -0.1, 0+ 1*block_h , phi, theta, psi],
            [0.15 + 2*x_offset -0.005  , -0.1, 0+ 2*block_h +0.005, phi, theta, psi],
            [0.15 + 3*x_offset  +0.01  , -0.1, 0+ 3*block_h +0.005 , phi, theta, psi], #-0.01
            [0.15 + 4*x_offset -0.005  , -0.14, 0+ 4*block_h, phi, theta, psi],
            [0.15 + 5*x_offset -0.012  , -0.14, 0+ 5*block_h -0.005, phi, theta, psi]
            # [0.15             , -0.1, 0, phi, theta, psi]
        ]
        # small destinations
        z_sm = -0.01 # - 1 cm
        dest_poses_sm = [
            [-0.25 - 5*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - 4*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - 3*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - 2*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - x_offset  , -0.05,  z_sm, phi, theta, psi],
            [-0.25             , -0.05,  z_sm, phi, theta, psi]
        ]


        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)


        index_lg = 0 # dest index for large blocks
        index_sm = 0
        Blocks = self.camera.blockDetector()
        Blocks.sort(key=my_custom_sort)
        while Blocks:
            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.065 # 6.5 cm
            if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
                block.wz = 0.027 # 2cm
            elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
                block.wz = 0.027 # 2cm
            else:
                block.wz = 0.032 # here
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm

            if sqrt(block.wx**2 + block.wy**2) >= 40.00/100:
                psi = radians(10)
            else:
                psi = radians(80)

            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))    
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            if dest_pose[2] >=  5*block_h:
                dest_z_offset = -0.02
            elif dest_pose[2] >= 4*block_h:
                dest_z_offset = -0.015
            elif dest_pose[2] >= 3*block_h:
                dest_z_offset = -0.007 # 0.6 cm lower
            elif dest_pose[2] >= 2*block_h:
                dest_z_offset = -0.008 # 0.8 cm lower
            elif dest_pose[2]>= 1*block_h:
                dest_z_offset = 0.005  # 0.5 cm higher
            else:
                dest_z_offset = 0.015  # 1cm higher
            
            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset

            # dest_joint wrist rotation:
            dest_pose[4] = 0 + (np.pi/2 - atan2(dest_pose[0], dest_pose[1]))

            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold

            df_wp2 = deepcopy(df_wp)
            # df_wp2[1] = radians(-np.pi/6)
            df_wp2[2] = radians(np.pi/6)
            df_wp2[3] = 0.0
            self.waypoints.append(df_wp2)
            self.replay_buffer.append(0) # hold

            df_wp3 = deepcopy(df_wp2)
            df_wp3[0] = radians(0)
            self.waypoints.append(df_wp3)
            self.replay_buffer.append(0) # hold



        print("Done event 4")
        self.next_state = "execute"


    def bonus_tothesky(self):

        self.current_state = "stack_em_up"
        # self.next_state = "execute"
        self.status_message = "Event4: Stack em up"
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []


        # First : Move ALL negative blocks to positive plane
        # block.wy <= 0.05 : move to 0.1

        homepose = np.array([0, 0, 0, 0, 0])
        # self.waypoints.append(homepose)
        # self.replay_buffer.append(1)
        self.rxarm.initialize()
        # self.rxarm.set_positions(list(homepose))
        # rospy.sleep(2)

        phi = 0.0
        psi = radians(80)



        print("Detecting blocks....")
        # while True:
        #     pass
            
        ### 1208 TODO : how to efficiently unstack? ######
        ########## Then, Unstack if needed ################

        """
            Unstack Logic :
            1. Use a for-loop for 4 levels
            2. Scan, if no block detected: go to lower level
            3. If block detected: move that block to its "left"
            4. repeat until we reach 1st level (stop)

            Expected z high:
                15 + 
                11 +
                7 +

            Note: use blockDetector_givenheight
        """

        # heights = [15.00, 11.00, 7.00]
        heights = [8.00, 5.00, 2.00]
        dirs = ["left", "right", "down"]

        bh = 30 # block height: 30,
        centers = [830,  866 , 904]
        widths = [20, 8, 8]


        # ## Relocate green and purple blocks!
        # self.rxarm.go_to_home_pose()
        # rospy.sleep(1)
        # self.rxarm.open_gripper()
        # rospy.sleep(1.5)
        # self.rxarm.set_positions(green_up_config)
        # rospy.sleep(2) 
        # self.rxarm.set_positions(green_far_config)
        # rospy.sleep(2)        
        # self.rxarm.close_gripper()
        # rospy.sleep(1.5)
        # self.rxarm.set_positions(green_up_config)
        # rospy.sleep(2) 
        # self.rxarm.set_positions(drop_green_config)
        # rospy.sleep(2)
        # self.rxarm.open_gripper()
        # rospy.sleep(1.5)


        # # self.rxarm.set_positions(purple_up_config)
        # # rospy.sleep(2) 
        # # self.rxarm.set_positions(purple_config)
        # # rospy.sleep(2)
        # # self.rxarm.close_gripper()
        # # rospy.sleep(1.5)
        # # self.rxarm.set_positions(purple_up_config)
        # # rospy.sleep(2) 
        # # self.rxarm.set_positions(drop_purple_config)
        # # rospy.sleep(2)
        # # self.rxarm.open_gripper()
        # # rospy.sleep(1.5)
        # self.rxarm.go_to_home_pose()
        # rospy.sleep(1.5)


        

        for i in range(3): ## Do 3 times (from level 4 to level 2 )
            
            Blocks = self.camera.blockDetector_givenheight(centers[i], widths[i])
            print("Unstacking level " +str(4-i))
            print("Detect "+str(len(Blocks))+ " blocks")
            if not Blocks:
                print("Empty block. Jump to lower level")
                continue
            while Blocks:
                block = Blocks.pop(0)
                if block.wz <= heights[i]/100 : # convert to meter
                    print("Too low, not this level so don't move.")
                else:
                    print("unstacking block "+block.color)
                    # Move this block to the dirs[i]
                    ######## For z offset ####
                    if sqrt(block.wx**2 + block.wy**2) <= 19.00/100:
                        z_offset = -0.06
                    elif sqrt(block.wx**2 + block.wy**2) <= 25.00/100:
                        z_offset = -0.04 # -3cm
                    else:
                        z_offset = +0.03 # 1cm

                    theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))
                    if block.wy >= 35.00/100:
                        psi = radians(10)
                        print("set pitch to 30 deg")
                    else:
                        psi = radians(80)
                    
                    if block.wz >= 0.03:
                        block.wz -= 0.03
                    # block.wx -= 0.005 # FOR THE ORANGE BLOCK!!!
                    # block.wy -= 0.005

                    block_position = np.array([block.wx, block.wy, block.wz + z_offset])
                    block_elbow_status = self.get_elbow_orientation(block_position)
                    block_pose = np.append(block_position, np.array([phi, theta, psi]))
                    up_pose= np.array([block.wx-0.1, block.wy-0.1, block.wz+ z_offset+0.1, phi, theta, psi])
                    up_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_pose), block_elbow_status, block.wz)
                    wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, block_pose), block_elbow_status, block.wz)
                    
                    dx = 0.08
                    dy = 0.08
                    z_offset = +0.04 #

                    if dirs[i] == "left":
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx - dx, block.wy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx - dx, block.wy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)
                        
                    elif dirs[i]=="down":
                        drop_pose = np.array([block.wx, block.wy - dy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx, block.wy - dy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                        
                    else: # right
                        # theta = theta + np.pi/2 # change WR 90 degree
                        drop_pose = np.array([block.wx + dx, block.wy, 0.0+ z_offset, phi, theta, psi])
                        drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, drop_pose), block_elbow_status, 0.0)
                        up_drop_pose= np.array([block.wx + dx, block.wy, 0.0+ z_offset+0.06, phi, theta, psi])
                        up_drop_wp =  self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, up_drop_pose), block_elbow_status, 0.0)
                        print("Drop at: ",drop_pose)                      

                    ########### Execute wps & gripper here ########            
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(wp)
                    rospy.sleep(2)
                    self.rxarm.close_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_wp)
                    rospy.sleep(2)

                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    self.rxarm.set_positions(drop_wp)
                    rospy.sleep(2)
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                    self.rxarm.set_positions(up_drop_wp)
                    rospy.sleep(2)
                    ###############################################
            self.rxarm.sleep()
            rospy.sleep(2)
        # return
        print("Done unsatcking")
        ###################################################

        print("Now stacking them high ...")

        # First, add a homepose and Open gripper
        homepose = np.array([0, 0, 0, 0, 0])
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        ########## Then, Line em up to the neg have plane ######
        x_offset = 0.015 # 6mm
        phi = 0.0
        theta=0.0
        block_h = 0.043 # 4cm
        psi = radians(80)
        dest_poses = [
            [0.15 + 0*x_offset         , -0.1, 0+ 0*block_h, phi, theta, psi],
            [0.15 + 1*x_offset         , -0.1, 0+ 1*block_h , phi, theta, psi],
            [0.15 + 2*x_offset -0.005  , -0.1, 0+ 2*block_h +0.005, phi, theta, psi],
            [0.15 + 3*x_offset -0.015  , -0.1, 0+ 3*block_h +0.005 , phi, theta, psi], #-0.01
            [0.15 + 4*x_offset -0.005  , -0.14, 0+ 4*block_h, phi, theta, psi],
            [0.15 + 5*x_offset -0.012  , -0.14, 0+ 5*block_h -0.005, phi, theta, psi]
            # [0.15             , -0.1, 0, phi, theta, psi]
        ]
        # small destinations
        z_sm = -0.01 # - 1 cm
        dest_poses_sm = [
            [-0.25 - 5*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - 4*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - 3*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - 2*x_offset, -0.05,  z_sm, phi, theta, psi],
            [-0.25 - x_offset  , -0.05,  z_sm, phi, theta, psi],
            [-0.25             , -0.05,  z_sm, phi, theta, psi]
        ]


        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)


        index_lg = 0 # dest index for large blocks
        index_sm = 0
        Blocks = self.camera.blockDetector()
        Blocks.sort(key=my_custom_sort)
        while Blocks:
            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # Offset for inter-waypoint 
            z_offset = 0.065 # 6.5 cm
            if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
                block.wz = 0.027 # 2cm
            elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
                block.wz = 0.027 # 2cm
            else:
                block.wz = 0.032 # here
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm

            if sqrt(block.wx**2 + block.wy**2) >= 40.00/100:
                psi = radians(10)
            else:
                psi = radians(80)

            theta = block.ori + (np.pi/2 - atan2(block.wx, block.wy))    
            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            if dest_pose[2] >=  5*block_h:
                dest_z_offset = -0.02
            elif dest_pose[2] >= 4*block_h:
                dest_z_offset = -0.015
            elif dest_pose[2] >= 3*block_h:
                dest_z_offset = -0.007 # 0.6 cm lower
            elif dest_pose[2] >= 2*block_h:
                dest_z_offset = -0.008 # 0.8 cm lower
            elif dest_pose[2]>= 1*block_h:
                dest_z_offset = 0.005  # 0.5 cm higher
            else:
                dest_z_offset = 0.015  # 1cm higher
            
            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset

            # dest_joint wrist rotation:
            dest_pose[4] = 0 + (np.pi/2 - atan2(dest_pose[0], dest_pose[1]))

            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold

            df_wp2 = deepcopy(df_wp)
            # df_wp2[1] = radians(-np.pi/6)
            df_wp2[2] = radians(np.pi/6)
            df_wp2[3] = 0.0
            self.waypoints.append(df_wp2)
            self.replay_buffer.append(0) # hold

            df_wp3 = deepcopy(df_wp2)
            df_wp3[0] = radians(0)
            self.waypoints.append(df_wp3)
            self.replay_buffer.append(0) # hold


        print("Done bonus event")
        self.next_state = "execute"


    ## First version
    def stack_em_high_original(self):

        self.current_state = "stack_em_high"
        self.status_message = "Event4 : stack_em_high"

        # Level1: R,G,B Large block
        """
            1. Move neg blocks to Pos plane
            2. Unstack all blocks
            3. Stack to a location at neg plane
        """
        
        # Clear all queue
        self.waypoints = []
        self.replay_buffer = []

        # self.rxarm.disable_torque()
        self.rxarm.enable_torque()
        homepose = np.array([0, 0, 0, 0, 0])

        """ Sweep method
            ############################################################
            print("Start sweeping......")
            self.rxarm.set_positions(homepose)
            rospy.sleep(1.5)
            # Sweep once to unstack all blocks
            sweep_wps = [
                [94,  0,  0,  0,   0],
                [94, 86.6, 88, 83, 1.4],
                [47, 86.6, 90, 83, 6.5],
                [-2, 86.6, 85, 86, 6.33],
                [-37, 86.6, 88, 86, 6.4],
                [-90, 86.6, 94, 86, 6.06],
                [-37, 86.6, 88, 86, 6.4]
            ]
            # leftmost_config = [91, 25, -48, 60, 3.8]
            # rightmost_config = [-91, 25, -48, 60, 3.8]
            for wp in sweep_wps:
                for i in range(len(wp)):
                    wp[i] = radians(wp[i])
                    # leftmost_config[i] = radians(leftmost_config[i])
                    # rightmost_config[i] = radians(rightmost_config[i])
                self.rxarm.set_positions(wp)
                rospy.sleep(2.5)
            self.rxarm.sleep()
            rospy.sleep(3)
            print("Complete sweeping")
            ############################################################
        """
        
        # First, add a homepose and Open gripper
        self.waypoints.append(homepose)
        self.replay_buffer.append(1)

        Blocks = self.camera.blockDetector()
        # x_offset = 0.1 # 12cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        # block_height = 0.0375
        # block_height_sm = 0.024 # 2.4cm
        block_height2 = 0.031
        x_offset2 = 0.0075 # 0.5cm
        z2 = -0.03
        dest_poses = [
            [0.2, -0.005,               z2, phi, theta, psi],
            [0.2  +x_offset2, -0.005,   z2 + block_height2, phi, theta, psi],
            [0.2  +2*x_offset2, -0.005, z2 +2*block_height2,  phi, theta, psi],
            [0.2  +3*x_offset2, -0.005, z2 +3*block_height2, phi, theta, psi],
            [0.2  +4*x_offset2, -0.005, z2 +4*block_height2, phi, theta, psi],
            [0.2  +5*x_offset2, -0.005, z2 +5*block_height2,  phi, theta, psi],
            [0.2  +6*x_offset2, -0.005, z2 +6*block_height2, phi, theta, psi],
            [0.2  +7*x_offset2, -0.005, z2 +7*block_height2,  phi, theta, psi]

        ]
        
        block_height_sm2 = 0.0185 # 2.4cm
        z_sm2 = -0.035 # - 1 cm
        x_offset_sm2 = 0.004 # 0.2cm
        dest_poses_sm = [
            [-0.22, -0.005, z_sm2, phi, theta, psi],
            [-0.22  -x_offset_sm2, -0.005, z_sm2 + block_height_sm2, phi, theta, psi],
            [-0.22  -2*x_offset_sm2, -0.005, z_sm2 +2*block_height_sm2,  phi, theta, psi],
            [-0.22  -3*x_offset_sm2, -0.005, z_sm2 +3*block_height_sm2, phi, theta, psi],
            [-0.22  -4*x_offset_sm2, -0.005, z_sm2 + 4*block_height_sm2, phi, theta, psi],
            [-0.22  -5*x_offset_sm2, -0.005, z_sm2 + 5*block_height_sm2,  phi, theta, psi],
            [-0.22  -6*x_offset_sm2, -0.005, z_sm2 + 6*block_height_sm2, phi, theta, psi],
            [-0.22  -7*x_offset_sm2, -0.005, z_sm2 + 7*block_height_sm2,  phi, theta, psi]
        ]


        def my_custom_sort(block):
            custom_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
            color = block.color
            if color in custom_order:
                return custom_order.index(color)
            else:
                return len(custom_order)

        # Sort Blocks in R,G,B order
        Blocks.sort(key=my_custom_sort)

        i = 0
        index_lg = 0 # dest index for large blocks
        index_sm = 0
        # Blocks is NOT empty
        while Blocks:
            # Pop out the first block
            block = Blocks.pop(0) 
            print("Doing block ", block.color, " size: ", block.size, " at (", block.wx, block.wy, block.wz, ")", " ori: ", block.ori)

            ### Calculate Pick Location IK
            # pose = np.array([block.wx, block.wy, block.wz, phi, block.ori, psi])

            # Offset for inter-waypoint 
            z_offset = 0.07 # 7cm

            if sqrt(block.wx**2 + block.wy**2) <= 20.00/100:
                if block.size=="large":
                    block.wz = 0.00 # 2cm
                else:
                    block.wz = -0.015 # -0.8cm
            elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
                if block.size=="large":
                    block.wz = 0.01 # 2cm
                else:
                    block.wz = 0.00 # -0.8cm
            else:
                block.wz = 0.032 # here
            
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm


            # If the block is too far, ignore it
            if sqrt(block.wx**2 + block.wy**2) >= 40.00/100:
                continue # Skip this block

            pose = np.array([block.wx, block.wy, block.wz + pick_z_offset, phi, theta, psi])
            first_pose =  np.array([block.wx, block.wy, block.wz+z_offset, phi, theta, psi])
            first_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz+z_offset])) # 0 is up, 1 is down
            first_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, first_pose), first_elbow_status, first_pose[2])
            block_elbow_status = self.get_elbow_orientation( np.array([block.wx, block.wy, block.wz]))
            block_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, pose), block_elbow_status, pose[2])

            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold
            self.waypoints.append(block_wp)
            self.replay_buffer.append(-1) # close
            self.waypoints.append(first_wp)
            self.replay_buffer.append(0) # hold

            ### Calculate Drop Location IK 
            if block.size == "large":
                dest_pose = np.asarray(dest_poses[index_lg])
                dest_first_pose = np.asarray(dest_poses[index_lg])
                index_lg += 1 # increment index to the next destination
            else:
                dest_pose = np.asarray(dest_poses_sm[index_sm])
                dest_first_pose = np.asarray(dest_poses_sm[index_sm])
                index_sm += 1

            # dest_z_offset = -0.008 # -0.8 cm
            if sqrt(dest_pose[0]**2 + dest_pose[1]**2) <= 18.00/100:
                dest_z_offset = -0.02 # -0.8 cm
            else:
                z_offset = 0.08       # interwaypoint offset: 8cm
                dest_z_offset = +0.02 # +1.0 cm

            dest_pose[2] += dest_z_offset
            dest_first_pose[2] += z_offset


            df_elbow_status = self.get_elbow_orientation( np.array([dest_first_pose[0],dest_first_pose[1],dest_first_pose[2]])) # 0 is up, 1 is down
            df_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_first_pose), df_elbow_status, dest_first_pose[2])
            dest_elbow_status = self.get_elbow_orientation( np.array([dest_pose[0], dest_pose[1], dest_pose[2]]))
            dest_wp = self.rxarm.find_best_soluton(IK_geometric(self.rxarm.dh_params, dest_pose), dest_elbow_status, dest_pose[2])

            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold            
            self.waypoints.append(dest_wp)
            self.replay_buffer.append(1) # open            
            self.waypoints.append(df_wp)
            self.replay_buffer.append(0) # hold
            
            # # Move to the next block:
            # i = i+1
 
            ### Execute wps & gripper here $####            
            index = 0
            for wp in self.waypoints:

                self.rxarm.set_positions(wp)
                rospy.sleep(2)
                if self.replay_buffer[index] == 0:
                    # just go to this waypoint; nothing required
                    pass
                elif self.replay_buffer[index] == 1: # open
                    self.rxarm.open_gripper()
                    rospy.sleep(1.5)
                else: # close gripper -1
                    self.rxarm.close_gripper()
                    rospy.sleep(1.5)

                # increment index
                index += 1

            # Clear all queue
            self.waypoints = [homepose]
            self.replay_buffer = [1] # Open

            self.rxarm.sleep()
            rospy.sleep(4)
            Blocks = self.camera.blockDetector()
            Blocks.sort(key=my_custom_sort)
            for block in Blocks:
                if block.wy <= 0.0:
                    Blocks.remove(block)
         
        print("Done event4")

        self.next_state = "execute"
        print("set next_state to execute")    


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)
