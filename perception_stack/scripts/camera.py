#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from capstone.msg import depthcam
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs

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
    
    
    def publish(self):
        ret, color_img, depth_img = self.next_frame()

        try:
            msg = depthcam()
            color_msg = self.bridge.cv2_to_imgmsg(color_img, "bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_img, "16UC1")

            color_msg.header.stamp  = rospy.Time.now()
            depth_msg.header.stamp  = color_msg.header.stamp
            msg.color = color_msg
            msg.depth = depth_msg
            self.camera_pub.publish(msg)

        except Exception as e:
            print(e)

    
    def release(self):
        self.pipeline.stop()


# def publish_color_and_depth(camera_pub, pipeline):
#     bridge = CvBridge()
#     try:
#         frames = pipeline.wait_for_frames()
#         align_to = rs.stream.color
#         align = rs.align(align_to)
#         aligned_frames = align.process(frames)
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         # frames = pipeline.wait_for_frames()
#         # color_frame = frames.get_color_frame()
#         # depth_frame = frames.get_depth_frame()

#         if not color_frame or not depth_frame:
#             return

#         msg = depthcam()
#         color_image = np.asanyarray(color_frame.get_data())
#         color_msg = bridge.cv2_to_imgmsg(color_image, "bgr8")

#         depth_image = np.asanyarray(depth_frame.get_data())
#         depth_image_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03) # adjust alpha value as needed
#         depth_image_8bit = cv2.normalize(depth_image_scaled, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
#         depth_msg = bridge.cv2_to_imgmsg(depth_image_8bit, "mono8")


#         color_msg.header.stamp  = rospy.Time.now()
#         depth_msg.header.stamp  = color_msg.header.stamp
#         msg.color = color_msg
#         msg.depth = depth_msg
#         camera_pub.publish(msg)


#     except Exception as e:
#         print(e)

def main():
    rospy.init_node("realsense_camera_node")
    
    camera = DepthCamera()
    while not rospy.is_shutdown():
        camera.publish()

    camera.release()

if __name__ == '__main__':
    main()