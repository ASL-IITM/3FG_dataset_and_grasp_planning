# FSR Main Package

The FSR Main package integrates force-sensitive resistors (FSRs) with a UR5 robot arm and a Robotiq 3-Finger Gripper using ROS (Robot Operating System). This README provides an overview of the package structure, functionality, and instructions for running the nodes.

## Package Contents

1. **Custom Message Definition**:
   - `FSRValues.msg`: Defines a custom ROS message type for publishing FSR values and corresponding force readings.

2. **ROS Nodes**:
   - `frc_slp_ctrl.py`: Controls the Robotiq 3-Finger Gripper based on FSR readings. Implements force control and slip detection logic.
   - `fsr_read.py`: Reads FSR data from a serial port, maps it to force values using polynomial functions derived from calibration data, and publishes the results.

3. **Launch File**:
   - `smart_grip.launch`: Launches all necessary nodes (`fsr_read.py`, `frc_slp_ctrl.py`, and `Robotiq3FGripperTcpNode.py`) to operate the system.

4. **Configuration Files**:
   - `CMakeLists.txt`: CMake build system configuration.
   - `package.xml`: Package manifest containing package dependencies and information.

5. **Source Files**:
   - `sheets`: Directory containing Excel sheets (`FSR_2.xlsx`, `FSR_3.xlsx`, `FSR_4.xlsx`) used for FSR calibration.

## Dependencies

- ROS (tested with Kinetic and Melodic)
- `robotiq_3f_gripper_articulated_msgs` package
- Python 3.x
- `numpy`, `pandas`, `matplotlib` Python libraries

## Running the Package

### Method 1: Individual Node Execution

1. Start ROS master node:
   ```bash
   roscore
   ```

2. Read FSR Data and Publish Force Values: Run `fsr_read.py` to read FSR data from a serial port and publish force values.

   ```bash
   rosrun fsr_main fsr_read.py
   ```

3. Control the Robotiq 3-Finger Gripper: Run `Robotiq3FGripperTcpNode.py` to control the Robotiq 3-Finger Gripper. Adjust the IP address as per your setup.

   ```bash
   rosrun robotiq_3f_gripper_control Robotiq3FGripperTcpNode.py 192.168.1.11
   ```

4. Implement Force Control and Slip Detection Logic: Run `frc_slp_ctrl.py` to implement force control and slip detection logic based on FSR readings.

   ```bash
   rosrun fsr_main frc_slp_ctrl.py
   ```

### Method 2:  Launch File Execution

1. Start ROS master node:
   ```bash
   roscore
   ```

2. Launch All Nodes Using `smart_grip.launch`: Launch the `smart_grip.launch` file to execute `fsr_read.py`, `Robotiq3FGripperTcpNode.py`, and `frc_slp_ctrl.py` simultaneously. This streamlines the setup process.

   ```bash
   roslaunch fsr_main smart_grip.launch
   ```
This launch file handles the coordinated execution of all nodes required for the FSR integration with the UR5 robot arm and Robotiq 3-Finger Gripper.

## Code Description

### FSR Data Reading and Publishing (`fsr_read.py`)

This script reads data from the FSR sensors and publishes it as a ROS message of type `FSRValues`.

#### Initialization:

- Initializes the ROS node and publisher.
- Reads calibration data from the Excel sheets to fit polynomial functions for force mapping.

#### Main Loop:

- Continuously reads FSR sensor data from the serial port.
- Maps the FSR values to force values using the polynomial functions.
- Publishes the FSR and force values as a ROS message.

### Force Control and Slip Detection (`frc_slp_ctrl.py`)

This script implements the force control and slip detection logic for the robotic gripper. It subscribes to the FSR values published by the `fsr_read.py` script and controls the gripper's position and force based on the real-time feedback.

#### Initialization:

- Initializes the ROS node and publishers/subscribers.
- Sets the desired force thresholds.
- Initializes various arrays and variables to track force and FSR values.

#### Main Loop:

- Continuously reads FSR values.
- Adjusts the gripper position and force based on the real-time feedback.
- Detects slip conditions and takes corrective actions.
  
## Acknowledgements

This project is developed to enhance the performance of a non-backdrivable robotic gripper by implementing advanced force control and slip detection mechanisms. Special thanks to the developers of the ROS and Robotiq 3F Gripper control package for their invaluable tools and support.

