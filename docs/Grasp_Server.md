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
