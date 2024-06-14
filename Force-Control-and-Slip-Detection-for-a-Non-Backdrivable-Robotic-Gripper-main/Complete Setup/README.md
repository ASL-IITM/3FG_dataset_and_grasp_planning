# Complete Setup for Calibrated FSR Integration with Robotiq Gripper

## Overview
This project involves integrating calibrated Force Sensitive Resistors (FSRs) with a Robotiq gripper, facilitated by a custom-designed PCB and ESP32 microcontroller. The setup includes CAD files for various components and code for data collection and communication. The system is designed to be compact and efficient, with a specific focus on ease of attachment to the UR5 robot arm.

## Folder Structure
- **CAD Files Fingertip**: Contains CAD files for the finger pad used to attach the FSRs to the Robotiq gripper.
- **PCB_Designs**: Contains files for the designed PCB, which incorporates the voltage divider circuit for compact system integration.
- **CAD Files Mounting Ring**: Contains CAD files for the 3D-printed ring used to attach the PCB and ESP32 to the UR5 robot arm.
- **FSR_Data_Collector.ino**: Arduino code for collecting data from the FSRs and handling serial communication.

## Setup Instructions

### Step 1: Assemble the Finger Pad
1. Open the `CAD Files Fingertip` folder and locate the CAD files for the finger pad.
2. 3D print the finger pad.
3. Attach the calibrated FSRs to the finger pad as shown below:
   
![FSR Attachment Diagram](https://github.com/JosephJoshyK/Force-Control-and-Slip-Detection-for-a-Non-Backdrivable-Robotic-Gripper/blob/main/Complete%20Setup/CAD%20Files%20Fingertip/Fingertip_cad_file.jpg)



### Step 2: Connect FSRs to ESP32
1. Connect the three FSRs to ports 2, 32, and 33 of the ESP32.
2. Ensure secure connections to avoid any signal issues.

### Step 3: Upload Code to ESP32
1. Locate `FSR_Data_Collector.ino`.
2. Open `FSR_Data_Collector.ino` in the Arduino IDE.
3. Upload the code to the ESP32. This code handles data collection from the FSRs and serial communication.

### Step 4: Design and Attach the PCB
1. Open the `PCB_Designs` folder and locate the PCB design files.
2. Fabricate the PCB according to the design files.
3. Determine the appropriate resistance for the FSRs to ensure a good range of output for the required force readings. Experiment with a potentiometer to find the best match.
4. Solder the necessary components to the PCB, ensuring the voltage divider circuits are correctly assembled.
5. Attach the PCB to the ESP32, ensuring all connections are secure.

### Step 5: Mount the System on the UR5
1. Open the `CAD Files Mounting Ring` folder and locate the CAD files for the 3D-printed ring.
2. 3D print the ring.
3. Attach the PCB and ESP32 to the UR5 robot arm using the 3D-printed ring as shown below:
   
   ![Mounting PCB and ESP32 on UR5 Robot Arm](https://github.com/JosephJoshyK/Force-Control-and-Slip-Detection-for-a-Non-Backdrivable-Robotic-Gripper/blob/main/Complete%20Setup/CAD%20Files%20Mounting%20ring/Complete%20Setup.png)


## Support
For any questions or issues, please contact josephkjoshy@gmail.com.

## Acknowledgements
- CAD models provided in `CAD Files Fingertip` and `CAD Files Mounting Ring`.
- PCB files are provided in `PCB_Designs`.
- Code developed for data collection and serial communication is provided in `FSR_Data_Collector.ino`.
