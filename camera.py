"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError

from math import radians, degrees

class Block():
    """!
    @brief      This class describes a block
    """
    def __init__(self, size_, color_, wx_, wy_, wz_, ori_):
        self.size = size_
        self.color = color_
        self.wx = wx_ # meter
        self.wy = wy_
        self.wz = wz_
        self.ori = ori_ # in radians

class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        self.BlockDetectFrame = np.zeros((720, 1280, 3)).astype(np.uint8)

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([[942.3094 , 0 , 637.3339] , [0 , 945.0566 , 357.1129] , [0 , 0 , 1]])
        self.extrinsic_matrix = np.array([[-1 , 0 , -0.025] , [0 , 1 , 0.160] , [0 , 0 , 1]])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.world_apriltag_coords = np.array([[-0.25 , -0.0025, 0.0] , [0.25 , -0.0025, 0.0] , [0.25 , 0.275, 0.0] , [-0.25, 0.275, 0.0]])
        self.dist_coeffs = np.array([0.122073 , -0.20644 , 0.003122 , 0.000990 , 0.000000])
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtBlockFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.BlockDetectFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    
    def transform_pixel_to_world(self, x, y):
        """
        @brief Transform point from pixel frame to world frame

        @param point: a coordinate in pixel frame

        @return a point in the world frame
        """
        point_3by1 = np.vstack((x, y , 1))

        A_inv = np.linalg.inv(self.intrinsic_matrix)
        pt_in_cam = np.matmul(A_inv , point_3by1)
        pt_in_world = np.matmul(np.linalg.inv(self.extrinsic_matrix) , pt_in_cam)
        pt_in_world =  np.delete(pt_in_world , 2)
        return pt_in_world


    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = list(( # in RGB order
            {'id': 'red', 'color': (220, 49, 56)}, #bgr 208
            {'id': 'orange', 'color': (255, 169, 35)},
            {'id': 'yellow', 'color': (255, 255, 27)},
            {'id': 'green', 'color': (60, 166, 100)},
            {'id': 'blue', 'color': (24, 132, 241)},
            {'id': 'violet', 'color': (110, 95, 198)}
        ))

        def retrieve_area_color(data, contour, labels):
            mask = np.zeros(data.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean = cv2.mean(data, mask=mask)[:3]
            # print("mean: ", mean)
            min_dist = (np.inf, None)
            for label in labels:
                d = np.linalg.norm(label["color"] - np.array(mean))
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
            return min_dist[1] 

        # Need arguments: 
        # image
        # lower
        # upper
        # depth - depth image
        print("In camera.blockDetector(): ")
        image = self.VideoFrame
        center = 940
        width = 20
        lower = center-width
        upper = center+width
        depth_image = self.DepthFrameRaw

        rgb_image = image
        cnt_image = image
        depth_data = depth_image
        # depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


        """mask out arm & outside board"""
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)

        cv2.rectangle(mask, (180,85),(1020,680), 255, cv2.FILLED)
        cv2.rectangle(mask, (510,79),(680,390), 0, cv2.FILLED)
        cv2.rectangle(cnt_image, (180,85),(1020,680), (255, 0, 0), 2)
        cv2.rectangle(cnt_image, (510,79),(680,390), (255, 0, 0), 2)
        # cv2.imshow("mask", mask)

        # threshold for depth:  thresh[i][j] = 1 if depth_data[i][j] within [lower, upper]
        # mask: Other than robot arm: All set to 1 (255 => 8bit: 11111111)
        # Only keeps detected area 1, others 0

        # print("shape", cv2.inRange(depth_data, lower, upper).shape)
        # print(cv2.inRange(depth_data, lower, upper))
        thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
        # cv2.imshow("Threshold window", thresh)

        kernel = np.ones((2,2), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closing ", closing)
        # closing = thresh

        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)
        print("Number of contours found: ", len(contours))


        # (cx, cy) is the (u,v) coord. in image frame
        # World_coord(X, Y, Z, 1) = ext * intrinsic * (cx, cy, 1)

        def img_to_world_coord(cx, cy):
            intrinsic_matrix = np.array([[942.3094 , 0 , 637.3339] , [0 , 945.0566 , 357.1129] , [0 , 0 , 1]])
            extrinsic_matrix = np.array([[-1 , 0 , -0.025] , [0 , 1 , 0.160] , [0 , 0 , 1]])
            A_inv = np.linalg.inv(intrinsic_matrix)
            pt_in_cam = np.matmul(A_inv , np.array([cx, cy, 1]))
            pt_in_world = np.matmul(np.linalg.inv(extrinsic_matrix) , pt_in_cam)
            return pt_in_world[0:3] # return x,y,z




        Blocks = []

        for contour in contours:
            # print("countour: ",contour)
            # print("cv2.minAreaRect(contour) ",cv2.minAreaRect(contour))
            color = retrieve_area_color(rgb_image, contour, colors)
            theta = cv2.minAreaRect(contour)[2]
            block_dim = cv2.minAreaRect(contour)[1]
            
            # smal block = 0 , large block = 1
            if (block_dim[0] < 40.0) and (block_dim[1] < 40.0):
                block_size = "small"
            else:
                block_size = "large"

            if block_dim[0] >= 50 or block_dim[0] >= 50 or block_dim[0]<=10 or block_dim[1]<=10:
                continue # too large or too small, don't append. It's outlier


            M = cv2.moments(contour)
            # division by zero if NOT found blocks
            if M['m00']==0:
                continue
            # else: NOT 0
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(cnt_image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
            cv2.putText(cnt_image, block_size, (cx-30, cy+60), font, 0.5, (255,0,0), thickness=2)
            cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)

            # wx, wy, wz = img_to_world_coord(cx, cy)
            wx, wy = self.transform_pixel_to_world(cx, cy)

            z = self.DepthFrameRaw[cy][cx]
            wz = (972.0 - z)/1000 


            # print(color, "theta(deg) "  ,int(theta), cx, cy, " world ", wx, wy, wz)
            

            Blocks.append(Block(block_size, color, wx, wy, wz, radians(theta)) )
            # Write cnt_image to self.BlockDetectImage
        
        self.BlockDetectFrame = cnt_image


        # TODO: return: Blocks (a list of block)
        # Each 'Block': an object that has:
        # size(large/small), color, (world x,y,z), orientation
        return Blocks


        # cv2.imshow("Image window", cnt_image)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()

        # pass





    def blockDetector_givenheight(self, center_, width_):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = list(( # in RGB order
            {'id': 'red', 'color': (220, 49, 56)}, #bgr 208
            {'id': 'orange', 'color': (255, 169, 35)},
            {'id': 'yellow', 'color': (255, 255, 27)},
            {'id': 'green', 'color': (60, 166, 100)},
            {'id': 'blue', 'color': (24, 132, 241)},
            {'id': 'violet', 'color': (110, 95, 198)}
        ))

        def retrieve_area_color(data, contour, labels):
            mask = np.zeros(data.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean = cv2.mean(data, mask=mask)[:3]
            # print("mean: ", mean)
            min_dist = (np.inf, None)
            for label in labels:
                d = np.linalg.norm(label["color"] - np.array(mean))
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
            return min_dist[1] 

        # Need arguments: 
        # image
        # lower
        # upper
        # depth - depth image
        print("In camera.blockDetector(): ")
        image = self.VideoFrame
        center = center_
        width = width_
        lower = center-width
        upper = center+width
        depth_image = self.DepthFrameRaw

        rgb_image = image
        cnt_image = image
        depth_data = depth_image
        # depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


        """mask out arm & outside board"""
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)

        cv2.rectangle(mask, (180,95),(1020,680), 255, cv2.FILLED)
        cv2.rectangle(mask, (502,85),(685,403), 0, cv2.FILLED)
        cv2.rectangle(cnt_image, (180,95),(1020,680), (255, 0, 0), 2)
        cv2.rectangle(cnt_image, (502,85),(685,403), (255, 0, 0), 2)
        # cv2.imshow("mask", mask)

        # threshold for depth:  thresh[i][j] = 1 if depth_data[i][j] within [lower, upper]
        # mask: Other than robot arm: All set to 1 (255 => 8bit: 11111111)
        # Only keeps detected area 1, others 0

        # print("shape", cv2.inRange(depth_data, lower, upper).shape)
        # print(cv2.inRange(depth_data, lower, upper))
        thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
        # cv2.imshow("Threshold window", thresh)

        kernel = np.ones((2,2), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closing ", closing)
        # closing = thresh

        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)
        print("Number of contours found: ", len(contours))


        # (cx, cy) is the (u,v) coord. in image frame
        # World_coord(X, Y, Z, 1) = ext * intrinsic * (cx, cy, 1)

        def img_to_world_coord(cx, cy):
            intrinsic_matrix = np.array([[942.3094 , 0 , 637.3339] , [0 , 945.0566 , 357.1129] , [0 , 0 , 1]])
            extrinsic_matrix = np.array([[-1 , 0 , -0.025] , [0 , 1 , 0.160] , [0 , 0 , 1]])
            A_inv = np.linalg.inv(intrinsic_matrix)
            pt_in_cam = np.matmul(A_inv , np.array([cx, cy, 1]))
            pt_in_world = np.matmul(np.linalg.inv(extrinsic_matrix) , pt_in_cam)
            return pt_in_world[0:3] # return x,y,z




        Blocks = []

        for contour in contours:
            # print("countour: ",contour)
            # print("cv2.minAreaRect(contour) ",cv2.minAreaRect(contour))
            color = retrieve_area_color(rgb_image, contour, colors)
            theta = cv2.minAreaRect(contour)[2]
            block_dim = cv2.minAreaRect(contour)[1]
            
            # smal block = 0 , large block = 1
            if (block_dim[0] < 40.0) and (block_dim[1] < 40.0):
                block_size = "small"
            else:
                block_size = "large"

            if block_dim[0] >= 50 or block_dim[0] >= 50 or block_dim[0]<=10 or block_dim[1]<=10:
                continue # too large or too small, don't append. It's outlier


            M = cv2.moments(contour)
            # division by zero if NOT found blocks
            if M['m00']==0:
                continue
            # else: NOT 0
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(cnt_image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
            cv2.putText(cnt_image, block_size, (cx-30, cy+60), font, 0.5, (255,0,0), thickness=2)
            cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)

            # wx, wy, wz = img_to_world_coord(cx, cy)
            wx, wy = self.transform_pixel_to_world(cx, cy)

            z = self.DepthFrameRaw[cy][cx]
            wz = (972.0 - z)/1000 


            # print(color, "theta(deg) "  ,int(theta), cx, cy, " world ", wx, wy, wz)

            Blocks.append(Block(block_size, color, wx, wy, wz, radians(theta)) )
            # Write cnt_image to self.BlockDetectImage
        
        self.BlockDetectFrame = cnt_image


        # TODO: return: Blocks (a list of block)
        # Each 'Block': an object that has:
        # size(large/small), color, (world x,y,z), orientation
        return Blocks


        # cv2.imshow("Image window", cnt_image)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()

        # pass



    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        # self.camera.BlockDetectFrame = self.camera.blockDetector()


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Block window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_frame = self.camera.convertQtBlockFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, block_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow(
                    "Block window",
                    cv2.cvtColor(self.camera.BlockFrame, cv2.COLOR_RGB2BGR))

                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
