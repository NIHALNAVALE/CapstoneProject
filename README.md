# Deep Learning - Driven Human Following Robot: Detection, Tracking and Navigation

### *RBE 594: Robotics Capstone Experience - Worcester Polytechnic Institute, Spring 2023*
### Members: Nihal Suneel Navale, Ashwij Kumbla, Himanshu Gautam, Purvang Patel

--------------------------------------------------------------

## Introduction
This project proposes a unique solution that incorporates a human to drive a robot autonomously
in any given environment. The main idea is to capture a desired person required to track and
follow, save particular features in the system. When the person enters the frame again, the feature
of this person is compared with the saved feature. Then the person is tracked to generate the
desired trajectories to follow. Finally, the motion planner is used to make the robot follow these
generated trajectories. The proposed solution is a framework that can be applied to ground-based
as well as aerial robots, however, considering the rigorous timeline, we have developed a
solution specific for ground-based robots.

## Requirements

Install the following dependencies when git cloning, there might be others that might not be mentioned here, be sure to install them.
```
  sudo apt-get install ros-noetic-amcl
```
```
  sudo apt-get install ros-noetic-gmapping
```
```
  sudo apt-get install ros-noetic-move-base
```

## Setup

Step1: Create a catkin workspace such as "catkin_ws" or "turtlebot_ws" and gitclone this repo.

*Steps 2,3 and 4 are to be launched in new terminal windows*

Step2: There are two launch files which you can use to launch the gazebo world, the first launch file launches a mapped world, while the second launches a unmapped world.   
```
  roslaunch turtlebot3_gazebo my_world.launch
```

```
  roslaunch turtlebot3_gazebo my_auto_nav.launch 
```

Step3: Navigate to "/CapstoneProject/src/turtlebot3_gazebo/scripts" and run the below command, this python file subcribes to the human pose message and publishes new calculated pose for the movebase package.
```
  python3 human_track.py
```

  
Step4: publish the pseudo-position of the human using the following rostopic pub, 
```
  rostopic pub /Human_Pose Perception/HumanPose "{depth: 2.0, bb_x: 1.0, bb_y: 0.0, frame_x: 0.0, frame_y: 0.0}"
```

depth is the position or depth the human is at from the camera/robot, bb_x is the distance the human is at from the center of the image frame.
