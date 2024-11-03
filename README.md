# Instructions to get started with the GraspIt! for Grasp Planning 

Host machine 
- OS version: Ubuntu 18.04
- ROS version: Melodic
- Create a Catkin workspace called catkin_ws under home  

- #### Note:  Currently we are using two seperate system to run moveit! and graspit! To be able to communicate properly, we need to have all the systems(UR5, moveit system, gripper, grasput system) to be on same network. Please refer to the Communication.md file for further information. 
## GraspIt! Installation instructions [GraspIt! Official Repository](https://github.com/graspit-simulator/graspit): 
### Install the following dependencies
```
sudo apt-get install libqt4-dev
sudo apt-get install libqt4-opengl-dev
sudo apt-get install libqt4-sql-psql
sudo apt-get install libcoin80-dev
sudo apt-get install libsoqt4-dev
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install libqhull-dev
sudo apt-get install libeigen3-dev
```
### Graspit Simulator setup
Follow the given steps to install the graspit simulator. Clone the repository in the home directory.
```
cd 
git clone https://github.com/graspit-simulator/graspit.git
cd graspit
mkdir build
cd build
cmake ..
make -j5
sudo make install
```
To ensure successful installation of Graspit, run the following command and check if the graspit simulator window pops up
```
 ~/graspit/build/graspit_simulator
```
### Graspit Support Packages
In this section, the procedure to build the packages graspit_commander and graspit_interface will be explained. Here it is assumed that a catkin workspace has been completely setup. The mentioned repositories should be cloned in the catkin_ws/src folder.

```
# Clone these repositories inside the src folder
cd ~/catkin_ws/src
git clone https://github.com/graspit-simulator/graspit_interface.git
git clone https://github.com/graspit-simulator/graspit_commander.git

# Build Workspace after cloning the packages
cd ~/catkin_ws
catkin_make

# After successful build of the packages
source devel/setup.bash
```
The graspit_interface package allows working with graspit simulator from ROS (Robotic Operating System), though only C++ language is supported. Incase of python support, the package graspit_commanander allows control over graspit simulator using python.

### The ROS command to launch the graspit_interface

<!-- Commands to run the interface -->
```
roslaunch graspt_interface graspit_interface.launch
```

Upon successful installation of Graspit, move onto the Script.md file to know how to use graspit interface through scripting.

# GraspIt! as a Grasp Planning Server

### This page provides information on how to setup graspit as a server for grasp planning.

The package **grasp_service_node** contains all the files that are needed to setup graspit as a server. 

First a custom service file called **pose_dof.srv** is created inside the folder srv. The contents of the file are as follows:
```
int32 goal_id # Used to find the object from the list of pre-existing primitive shapes.

int32 counter # A counter variable to keep track of the number of times the server is accessed.

geometry_msgs/Pose obj_pose # The pose of the object wrt the world frame obtained from the perception module.
---
geometry_msgs/Pose gripper_pose # The pose of the gripper wrt the world frame sent back to the client after grasp planning.

float32[] dofs # Joint angles of all the fingers of the RobotIQ 3-Finger gripper.
```

Once the custom service file is created, the CMakeLists.txt is updated with the information of the new service files.

### Running the Grasp_Server

To run the grasp server, run the following commands in terminal:
```
roslaunch graspit_interface graspit_interface.launch
```

Once the graspit_window has launched, in a new terminal window, run the following command:

```
rosrun grasp_service_node full_grasp.py 
```

# GraspIt! Scripting using graspit_commander

## Gripper Package

**Assuming the gripper package is provided along with the repository**

The script grasp.py file provided inside the folder scripts of the gripper package performs Eigen Grasp Planning on a cube (can be changed in script). To run the script do the following.
```
# Launch the graspit interface window
roslaunch graspit_interface graspit_interface.launch

# In a new terminal window
rosrun gripper grasp.py
```

The grasp.py executable does grasp planning using a Robotiq gripper and a cylinder for 70,000 iterations.

For using GraspIt! as a server, please have a look at the file Grasp_Server.md
