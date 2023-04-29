#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from capstone.msg import HumanPose
from geometry_msgs.msg import Twist


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp  # proportional gain
        self.Ki = Ki  # integral gain
        self.Kd = Kd  # derivative gain
        self.last_error = 0  # initialize the last error term
        self.integral = 0  # initialize the integral term
        self.msg = Twist()
        self.robot_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    # def update(self, setpoint, feedback, dt=0.2):
    #     error = setpoint - feedback  # calculate the error term
    #     self.integral += error * dt  # calculate the integral term
    #     derivative = (error - self.last_error) / dt  # calculate the derivative term
    #     output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative  # calculate the output
    #     self.last_error = error  # store the current error term for the next iteration
    #     return output

    def callback(self,data):
        rot_error = data.bb_x - data.frame_x
        print(rot_error)
        self.msg.angular.z = -self.Kp * rot_error
        self.msg.linear.x = self.Kd * (data.depth - 1.5)
        # print(data.depth -1.5)
        # # print(self.Kd * data.depth - 1.5)

        self.robot_pub.publish(self.msg)
        




def main():

    # Initialize the ROS node
    rospy.init_node('robot_Controller')

    controller = PIDController(0.003, 0.1, 0.2)
    
    color_sub = rospy.Subscriber('/Human_pose', HumanPose, controller.callback)

    rospy.spin()

    # Clean up the OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()