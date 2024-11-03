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