#!/usr/bin/env python3

from __future__ import print_function
import rospy
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotOutput
# from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotInput
import numpy as np
from fsr_main.msg import FSRValues
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import time

# Initialize ROS node
rospy.init_node('gripper_control_logic', anonymous=True)

# Set the desired force threshold
FORCE_THRESHOLD_MIN = 15
FORCE_THRESHOLD_MAX = 48
init_time = 1

# Publisher for Robotiq3FGripperRobotOutput messages
pub = rospy.Publisher("Robotiq3FGripperRobotOutput", Robotiq3FGripperRobotOutput, queue_size=10)

# Initial command setup
command = Robotiq3FGripperRobotOutput()
command.rACT = 1
command.rGTO = 1
command.rMOD = 0
command.rPRA = 0
command.rFRA = 50
command.rSPA = 255
pub.publish(command)

# Initialize arrays and variables
force_v = []
force_w = []
force_x = []

fsr_v = 0
fsr_w = 0
fsr_x = 0
force_v = 0
force_w = 0
force_x = 0

v = []
w = []
x = []

value_stack_v = []
value_stack_w = []
value_stack_x = []

# Initialize lists to store incoming values
fsr_v_val = []
fsr_w_val = []
fsr_x_val = []

decrease_detected_v = False
decrease_detected_w = False
decrease_detected_x = False

previous_value_v = 0
previous_value_w = 0
previous_value_x = 0

# Define global variables to track force thresholds
force_v_reached_threshold = False
force_w_reached_threshold = False
force_x_reached_threshold = False

def detect_slip(values):
    """
    Detect slip based on the values provided.

    Args:
        values (list): List of values to check for slip.

    Returns:
        bool: True if slip is detected, False otherwise.
    """
    min_value = min(values)
    negative_values = [value for value in values if value < 0 and value != min_value]

    if len(negative_values) >= 1:
        average_negative = sum(negative_values) / len(negative_values)
    else:
        average_negative = -30

    slip_detected = min_value < 10 * average_negative
    return slip_detected

def fsr_values_callback(msg):
    """
    Callback function to process FSR values from the subscribed topic.

    Args:
        msg (FSRValues): The message containing FSR values.
    """
    global fsr_v, fsr_w, force_v, fsr_x, force_w, force_x
    fsr_v = msg.fsr_v
    fsr_w = msg.fsr_w
    fsr_x = msg.fsr_x
    force_v = msg.force_v
    force_w = msg.force_w
    force_x = msg.force_x

def update_value_stack(stack, value):
    """
    Update the stack of values by adding the new value and removing the oldest value if the stack size exceeds 5.

    Args:
        stack (list): The stack to update.
        value (int): The new value to add.
    """
    stack.append(value)

    if len(stack) > 5:
        stack.pop(0)

def p_control(force_value, force_threshold_min, kp=0.5):
    """
    Proportional control function to calculate control output.

    Args:
        force_value (float): The current force value.
        force_threshold_min (float): The minimum force threshold.
        kp (float): The proportional gain.

    Returns:
        int: The control output.
    """
    error = force_threshold_min - force_value
    control_output = kp * error
    control_output = int(Decimal(control_output).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

    # Ensure the control output is within the desired range (1 to 3)
    add_val = max(1, min(3, control_output))
    
    return add_val

def increment_position(command, add_val):
    """
    Increment the gripper position.

    Args:
        command (Robotiq3FGripperRobotOutput): The command to update.
        add_val (int): The value to add to the position.

    Returns:
        Robotiq3FGripperRobotOutput: The updated command.
    """
    command.rPRA += add_val
    return command

def decrement_position(command, sub_val):
    """
    Decrement the gripper position.

    Args:
        command (Robotiq3FGripperRobotOutput): The command to update.
        sub_val (int): The value to subtract from the position.

    Returns:
        Robotiq3FGripperRobotOutput: The updated command.
    """
    command.rPRA -= sub_val
    return command

def increment_force(command, add_val):
    """
    Increment the gripper force.

    Args:
        command (Robotiq3FGripperRobotOutput): The command to update.
        add_val (int): The value to add to the force.

    Returns:
        Robotiq3FGripperRobotOutput: The updated command.
    """
    command.rFRA += add_val
    return command

def decrement_force(command, sub_val):
    """
    Decrement the gripper force.

    Args:
        command (Robotiq3FGripperRobotOutput): The command to update.
        sub_val (int): The value to subtract from the force.

    Returns:
        Robotiq3FGripperRobotOutput: The updated command.
    """
    if command.rFRA > 50:
        command.rFRA -= sub_val
    
    return command

def main():
    """
    Main function to control the gripper based on FSR values and slip detection.
    """
    pub = rospy.Publisher("Robotiq3FGripperRobotOutput", Robotiq3FGripperRobotOutput, queue_size=10)
    fsr_sub = rospy.Subscriber("/fsr_values", FSRValues, fsr_values_callback)
    global force_v, force_w, force_x, fsr_v, fsr_w, fsr_x, command
    global init_time

    # Append values to the lists
    fsr_v_val.append(fsr_v)
    fsr_w_val.append(fsr_w)
    fsr_x_val.append(fsr_x)
    
    update_value_stack(value_stack_v, fsr_v)
    update_value_stack(value_stack_w, fsr_w)
    update_value_stack(value_stack_x, fsr_x)

    try:
        while not rospy.is_shutdown():      
            # Proportional control to increase the gripper position if the force is below the minimum threshold
            while (int(force_v < FORCE_THRESHOLD_MIN) + int(force_w < FORCE_THRESHOLD_MIN) + int(force_x < FORCE_THRESHOLD_MIN)) == 3 and command.rPRA < 253:
                add = p_control(force_v, FORCE_THRESHOLD_MIN)
                command = increment_position(command, add)
                pub.publish(command)
                rospy.sleep(0.1)

                # Append values to the lists
                fsr_v_val.append(fsr_v)
                fsr_w_val.append(fsr_w)
                fsr_x_val.append(fsr_x)

                update_value_stack(value_stack_v, fsr_v)
                update_value_stack(value_stack_w, fsr_w)
                update_value_stack(value_stack_x, fsr_x)

                init_time += 1

            # Decrease the gripper position and force if the force exceeds the maximum threshold
            while (int(force_v > FORCE_THRESHOLD_MAX) + int(force_w > FORCE_THRESHOLD_MAX) + int(force_x > FORCE_THRESHOLD_MAX)) > 0 and command.rPRA > 2 and command.rFRA > 24:
                command = decrement_position(command, 3)
                pub.publish(command)
                rospy.sleep(0.1)
                command = decrement_force(command, 25)            
                pub.publish(command)
                rospy.sleep(0.1)
                
                fsr_v_val.append(fsr_v)
                fsr_w_val.append(fsr_w)
                fsr_x_val.append(fsr_x)

                update_value_stack(value_stack_v, fsr_v)
                update_value_stack(value_stack_w, fsr_w)
                update_value_stack(value_stack_x, fsr_x)
            
            fsr_v_val.append(fsr_v)
            fsr_w_val.append(fsr_w)
            fsr_x_val.append(fsr_x)

            update_value_stack(value_stack_v, fsr_v)
            update_value_stack(value_stack_w, fsr_w)
            update_value_stack(value_stack_x, fsr_x)

            gradient_v = np.diff(value_stack_v)
            gradient_w = np.diff(value_stack_w)
            gradient_x = np.diff(value_stack_x)

            slip_detected_v = detect_slip(gradient_v)
            slip_detected_w = detect_slip(gradient_w)
            slip_detected_x = detect_slip(gradient_x)

            # If slip is detected, print the time
            if sum([slip_detected_v, slip_detected_w, slip_detected_x]) >= 2:
                current_time = datetime.now()
                formatted_time = current_time.strftime("%H:%M:%S")
                print('Slip detected! at:', formatted_time)

            init_time += 1   
            # Add a small delay if needed to control the data acquisition rate
            # time.sleep(0.1)

    except rospy.ROSInterruptException:
        pass
    finally:
        # Cleanup code
        print("Shutting down ROS node")
        command = Robotiq3FGripperRobotOutput()
        command.rACT = 0
        pub.publish(command)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
