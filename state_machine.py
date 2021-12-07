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
from math import radians, sqrt
from copy import deepcopy


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
        self.status_message = "State: Execute - Executing motion plan"
        self.next_state = "idle"
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
        x_offset = 0.1 # 12cm
        phi = 0.0
        theta=0.0
        psi = radians(80)

        dest_poses = [
            [0.2 + 2*x_offset, -0.005, 0, phi, theta, psi],
            [0.2 + x_offset, -0.005, 0, phi, theta, psi],
            [0.2, -0.005, 0 ,  phi, theta, psi],
            [0.2 + 2*x_offset, -0.075, 0, phi, theta, psi],
            [0.2 + x_offset, -0.075, 0, phi, theta, psi],
            [0.2, -0.075, 0 ,  phi, theta, psi]
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


    def pick_n_stack(self):

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
        block_height0 = 0.04
        block_height1 = 0.035
        block_height2 = 0.0275
        x_offset0 = 0.012
        x_offset1 = 0.008
        x_offset2 = 0.0075 # 0.5cm
        z0 = 0.008 # 0.8 cm
        z1 = -0.005
        z2 = -0.03
        dest_poses = [
            [0.4, -0.005, z0, phi, theta, psi],
            [0.4+  x_offset0, -0.005, z0 + block_height0, phi, theta, psi],
            [0.4+2*x_offset0, -0.005, z0 +2*block_height0+0.006,  phi, theta, psi],
            [0.3, -0.005, z1, phi, theta, psi],
            [0.3+x_offset1, -0.005, z1 + block_height1, phi, theta, psi],
            [0.3+2*x_offset1, -0.005, z1 +2*block_height1+0.002,  phi, theta, psi],
            [0.18, -0.005, z2, phi, theta, psi],
            [0.18+x_offset2, -0.005, z2 + block_height2, phi, theta, psi],
            [0.18+2*x_offset2, -0.005, z2 +2*block_height2+0.002,  phi, theta, psi]
        ]
        
        block_height_sm0 = 0.025 # 2.4cm        
        block_height_sm1 = 0.02 # 2.4cm
        block_height_sm2 = 0.0185 # 2.4cm
        # small destinations
        z_sm0 =  0.00
        z_sm1 = -0.01
        z_sm2 = -0.35 # - 1 cm
        x_offset_sm0 = 0.005
        x_offset_sm1 = 0.0018
        x_offset_sm2 = 0.002 # 0.2cm
        # dest_poses_sm = [
        #     [-0.2, -0.005, z_sm, phi, theta, psi],
        #     [-0.2-x_offset_sm, -0.005, z_sm + block_height_sm, phi, theta, psi],
        #     [-0.2-2*x_offset_sm, -0.005, z_sm+2*block_height_sm,  phi, theta, psi]
        # ]
        # level3:
        dest_poses_sm = [
            [-0.4, -0.005, z_sm0, phi, theta, psi],
            [-0.4-x_offset_sm0, -0.005, z_sm0 + block_height_sm0, phi, theta, psi],
            [-0.4-2*x_offset_sm0, -0.005, z_sm0+2*block_height_sm0+0.002,  phi, theta, psi],
            [-0.3, -0.005, z_sm1, phi, theta, psi],
            [-0.3-x_offset_sm1, -0.005, z_sm1 + block_height_sm1, phi, theta, psi],
            [-0.3-2*x_offset_sm1, -0.005, z_sm1+2*block_height_sm1,  phi, theta, psi],
            [-0.18, -0.005, z_sm2, phi, theta, psi],
            [-0.18-x_offset_sm2, -0.005, z_sm2 + block_height_sm2, phi, theta, psi],
            [-0.18-2*x_offset_sm2, -0.005, z_sm2+2*block_height_sm2,  phi, theta, psi]
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
                    block.wz = -0.008 # -0.8cm
            elif sqrt(block.wx**2 + block.wy**2) <= 28.00/100:
                block.wz = 0.01 # 2cm
            else:
                block.wz = 0.032 # here
            
            # Pick z offset depends on size
            if block.size == "large":
                pick_z_offset = -0.015 # -1.5cm
            else:
                pick_z_offset = -0.02 # -2cm

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
             
        print("Done event2")

    def line_em_up(self):


        reset_joint_config = np.array([-1.67, 0,  0, 0, 0])

        # After placing each block, set_joint_position(reset_joint_config)
        # and run blockDetector again

        Blocks = self.camera.blockDetector()

        while Blocks:


            # After placing:
            self.rxarm.set_positions(reset_joint_config)
            Blocks = self.camera.blockDetector()


            pass


        pass

    def stack_em_high(self):

        pass

    


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
