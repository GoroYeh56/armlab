"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2

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
            rospy.sleep(3.5)
            if self.replay_buffer[index] == 0:
                # just go to this waypoint; nothing required
                pass
            elif self.replay_buffer[index] == 1: # open
                self.rxarm.open_gripper()
                rospy.sleep(1)
            else: # close gripper
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
            print('errors: ', total_errors)

            # increment index
            index += 1

        print("max errors in each waypoint: ", max_errors)
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


        print("pt_world: " , pt_world)
        print("pt_camera: " , pt_camera)
        print("pt_image: ", pt_image_T.T)

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

        print("Rot: ", Rot)
        print("Rot_3by3: ", Rot_3by3)
        print("trans: ", trans)

        # assemble extrinsic matrix
        bot_row = np.array([0 , 0 , 0 , 1])
        self.camera.extrinsic_matrix = np.vstack((np.hstack((Rot_3by3 , trans)) , bot_row))

        print("EXTRINSIC: " , self.camera.extrinsic_matrix)
        
        # Add 1 to each pt in world_apriltag_coords   
        ones = np.array([1, 1, 1, 1]).T.reshape(-1,1)
        pt_world = np.hstack(( pt_world, ones)) 

        # verify accuracy of extrinsic matrix
        pt_camera_verified = np.matmul(self.camera.extrinsic_matrix, pt_world.T)
        print("Ours pt_camera: ", pt_camera_verified.T) # row by row
        print("pt_camera from aprilTag: ", pt_camera)
        
        
        rot_part = self.camera.extrinsic_matrix[0:2, 0:2]
        trans_part = self.camera.extrinsic_matrix[0:2, 3]
        print("rot_part: ", rot_part)
        print("trans_part: ", trans_part)
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
