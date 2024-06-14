# Instructions on controlling 3 finger gripper using ROS

The [ros-industrial repository](https://github.com/ros-industrial/robotiq) is unmaintained by robotiq and last distro support it has is of kinetic branch.

Another repository of [3 finger Robotiq gripper](https://github.com/TAMS-Group/robotiq) which is fork of ROS industrial is actively maintained by TAMS group from University of Hamburg and has the support of noetic distro.  

We want to control 3 finger gripper using Modbus TCP protocol  
Assuming you have cloned the repository mentioned above.

## Following instructions given on [3 finger robotiq gripper](http://wiki.ros.org/robotiq/Tutorials/Control%20of%20a%203-Finger%20Gripper%20using%20the%20Modbus%20TCP%20Protocol)

- The Gripper should be connected to a network, which has been properly configured (you can validate the communication with the Gripper using the [Robotiq User Interface](https://robotiq.com/support)). Make sure that external dependency of package "robotiq_modbus_tcp" is installed. The dependency is the python package pyModbus.

- Network configuration: Make sure that PC connected to gripper has static IP address different from default IP address of gripper which is 192.168.1.11

### Using ROS nodes to control gripper

- The Gripper is driven by the node "Robotiq3FGripperTcpNode.py" contained in the package "robotiq_3f_gripper_control". 
```
# change into the ur_ws workspace

cd ~/catkin_ws

# run the 3f gripper node, the IP address used here is the default IP address of gripper

rosrun robotiq_3f_gripper_control Robotiq3FGripperTcpNode.py 192.168.1.11
```
The driver listens for messages on "Robotiq3FGripperRobotOutput" using the "Robotiq3FGripper_robot_output" msg type. The messages are interpreted and commands are sent to the gripper accordingly

- In another terminal run a simple controller node 

```
rosrun robotiq_3f_gripper_control Robotiq3FGripperSimpleController.py
```
The simple controller node can be modified to send custom commands to the Gripper. 
```
'''here are some custom commands added to the Robotiq3FGripperSimpleController.py to operate each finger seperately'''

    if char == "fa_on":
        command.rICF = 1
        command.rPRA = 255

    if char == "fa_off":
        command.rPRA = 0
        command.rICF = 0        
       
    if char == "fb_on":
        command.rICF = 1
        command.rPRB = 255
    
    if char == "fb_off":
        command.rPRB = 0
        command.rICF = 0        
    
    if char == "fc_on":
        command.rICF = 1
        command.rPRC = 255

    if char == "fc_off":
        command.rPRC = 0
        command.rICF = 0        

```
- The status listener node can be run at another terminal.The driver publishes the status of the Gripper on "Robotiq3FGripperRobotInput" using the "Robotiq3FGripper_robot_input" msg type. 
```
rosrun robotiq_3f_gripper_control Robotiq3FGripperStatusListener.py
