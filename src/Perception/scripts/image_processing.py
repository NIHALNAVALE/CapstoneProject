#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from capstone.msg import depthcam,HumanPose
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs

from Human_Detection import *
from FeatureExtraction import *
from gg import *
from utils.datasets import letterbox

class DepthCamera:
    def __init__(self,frame_rate=30):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()
        self.camera_ON =False
        config = rs.config()
        config.enable_stream(rs.stream.color, rs.format.bgr8, frame_rate)
        config.enable_stream(rs.stream.depth, rs.format.z16, frame_rate)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.bridge = CvBridge()

        self.camera_pub = rospy.Publisher("/camera", depthcam, queue_size=10)


    def next_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None
        
        color_image = np.asanyarray(color_frame.get_data())

        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        depth_image = np.asanyarray(filled_depth.get_data())
        # depth_image_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03) # adjust alpha value as needed
        # depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        return True, color_image, depth_image

class IDK:
    
    def __init__(self):
        self.ConfigFlag = False
        self.camera = DepthCamera()
        self.bridge = CvBridge()
        self.H_detector = Human_Detection()
        
        self.fe = SuperPointFrontend(nms_dist=4,conf_thresh=0.015,nn_thresh=0.7)
        self.saved_features_to_track = np.empty((256, 1))

        self.pose_pub = rospy.Publisher("/Human_pose", HumanPose, queue_size=10)
        
    def CallBack(self):
        if(self.ConfigFlag == False):
            self.RobotConfigCallback()
        
        else:
            self.RobotRunCallback()

    def RobotConfigCallback(self):
        ret,rgb_frame, depth_frame = self.camera.next_frame()
        cv_image_color = rgb_frame
        # cv_image_depth = self.bridge.imgmsg_to_cv2(data.depth, "mono8")

        frame = letterbox(cv_image_color, 640, stride=64, auto=True)[0] 
        conf_box = [int(frame.shape[1]/2)-120,0,
                           int(frame.shape[1]/2)+120,frame.shape[0]]
        
        detection = self.H_detector.detect(frame)

        if(detection == None):
            return 
        
        track_box,track_mask = PersonToTrack(detection, conf_box)
        if(track_box is None or track_mask is None):
            return
        
        person_to_track = CropnMask(frame,track_box,track_mask)
        person_to_track_grey = cv2.cvtColor(person_to_track,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        pts,desc, _ = self.fe.run(person_to_track_grey)
        self.saved_features_to_track =  np.concatenate((self.saved_features_to_track, desc), axis=1)

        cv2.imshow("person_to_track",person_to_track)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.ConfigFlag = True
            cv2.destroyAllWindows()
    
    def RobotRunCallback(self):
        ret,cv_image_color, cv_image_depth = self.camera.next_frame()
        # cv_image_color = self.bridge.imgmsg_to_cv2(data.color, "bgr8")
        # cv_image_depth = self.bridge.imgmsg_to_cv2(data.depth, "16UC1")
        
        frame = letterbox(cv_image_color, 640, stride=64, auto=True)[0]
        depth = letterbox(cv_image_depth, 640, stride=64, auto=True)[0]

        detection = self.H_detector.detect(frame)

        if(detection == None):
            return 
        
        best_matches_count = 0
        for one_mask, bbox, _, _ in detection:
            
            draw(frame, bbox, one_mask, color=(255,0,0))
            person = CropnMask(frame,bbox,one_mask)
            person_grey = cv2.cvtColor(person,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            pts, desc, _ = self.fe.run(person_grey)

            good_pt_idx = np.where(pts[2,:] > np.median(pts[2,:]))[0]
            good_features = pts[:,good_pt_idx]
            good_descriptors = desc[:,good_pt_idx] 


            feature_matches = self.fe.nn_match_two_way(self.saved_features_to_track, desc,0.8)
            # feature_matches = nn_match_two_way(saved_features_to_track, good_descriptors)

            

            if(feature_matches.shape[1] > best_matches_count):
                best_matches_count = feature_matches.shape[1]
                best_matches = feature_matches
                best_mask, best_box = one_mask, bbox

        # print(best_matches_count)
        # print(np.mean(best_matches[2,:]),np.median(best_matches[2,:]))
        if best_matches_count > 35:
            bb_centre_x = int((best_box[0]+best_box[2])/2)
            bb_centre_y = int((best_box[1]+best_box[3])/2)
            cv2.circle(frame, (bb_centre_x,bb_centre_y), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), radius=10, color=(0, 0, 0), thickness=-1)
            dist_mm = depth[bb_centre_y,bb_centre_x] 
            cv2.putText(frame, "{} m".format(dist_mm/1000), (best_box[0] + 5, best_box[1] + 60), 0, 1.0, (255, 255, 255), 2)
            draw(frame, best_box, best_mask, color=(0,255,0))
            
            msg = HumanPose()
            msg.depth = dist_mm/1000
            
            msg.bb_x = bb_centre_x 
            msg.bb_y = bb_centre_y 
            msg.frame_x = int(frame.shape[1]/2)
            msg.frame_y = int(frame.shape[0]/2)
            self.pose_pub.publish(msg)

        cv2.imshow("detection_frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):

            cv2.destroyAllWindows()
            







def main():

    # Initialize the ROS node
    rospy.init_node('perception_stack')
    t = IDK()
    while not rospy.is_shutdown():
        t.CallBack()

    
    # Set up the ROS spin loop
    rospy.spin()

    # Clean up the OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()