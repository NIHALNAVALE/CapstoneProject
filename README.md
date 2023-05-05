# Deep Learning - Driven Human Following Robot: Detection, Tracking and Navigation

### *RBE 594: Robotics Capstone Experience - Worcester Polytechnic Institute, Spring 2023*
### Members: Nihal Suneel Navale, Ashwij Kumbla, Himanshu Gautam, Purvang Patel

Step1: Create a catkin workspace such as "catkin_ws" or "turtlebot_ws" and gitclone this repo.

Steps 2,3 and 4 are to be launched in new terminal windows

Step2: roslaunch turtlebot3_gazebo my_world.launch 
       for pre-mapped world.
       roslaunch turtlebot3_gazebo my_auto_nav.launch
       for SLAM (no-pre mapped world). 
Step3: Navigate to "/CapstoneProject/src/turtlebot3_gazebo/scripts"
       and run python3 human_track.py
  
Step4: publish the pseudo-position of the human using the following rostopic pub, $ rostopic pub /Human_Pose Perception/HumanPose "{depth: 2.0, bb_x: 1.0, bb_y: 0.0, frame_x: 0.0, frame_y: 0.0}"

depth is the position or depth the human is at from the camera/robot, bb_x is the distance the human is at from the center of the image frame.
