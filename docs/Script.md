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