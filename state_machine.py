"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy

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
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]

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

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
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

        # send kinematics
        print('executing...')
        max_errors = {}
        for wp in waypoints:
            self.rxarm.set_positions(wp)
            print(wp)
            # sleep to get to waypoint\
            rospy.sleep(3)
            # print("error: " , list(self.rxarm.get_joint_positions()) - wp )
            total_errors = []
            for i in range(len(wp)):
                diff = abs(self.rxarm.get_joint_positions()[i] - wp[i])
                total_errors.append(diff)
            max_errors[self.rxarm.joint_names[np.argmax(total_errors)]] = max(total_errors)
            print('errors: ', total_errors)

        print("max errors in each waypoint: ", max_errors)
        self.status_message = "State: Execute - Executing motion plan"
        self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        print("enter calibrate state")


        pt_world = self.camera.world_apriltag_coords
        # pt_camera = np.matmul(np.linalg.inv(self.camera.intrinsic_matrix), np.asarray(pt_pixel))
        pt_camera = []
        ext_matrices = []
        for i in range(len(pt_world)) :
            x = self.camera.tag_detections.detections[i].pose.pose.pose.position.x
            y = self.camera.tag_detections.detections[i].pose.pose.pose.position.y
            z = self.camera.tag_detections.detections[i].pose.pose.pose.position.z                        
            pt_camera.append( np.array([x,y, z, 1]) )
            # pt_camera.append(np.array([x,y]))
            # print("pt camera[i] ", pt_camera[i])
            pt_world[i] = np.asarray(pt_world[i])
            # print("pt_world[i] asarray ", np.asarray(pt_world[i]))

        ### 11/11 TODO : SolvePnP()

        print("pt_camera: ", pt_camera)
        print("pt_world: ",pt_world)
        # ext_matrices.append( self.camera.getAffineTransform( np.asarray(pt_camera) , np.asarray(pt_world)) )
        
        ext = np.matmul ( np.asarray(pt_world).T , np.linalg.inv(np.asarray(pt_camera).T) )
        print("extrinsic matrix:", ext)
        for i in range(len(pt_world)):
            print("pt ", i+1, " camera: ", pt_camera[i], " => world: ", np.matmul(ext, pt_camera[i]))

        # print("extrinsic_matrix: ", ext_matrices)
        
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