#!/usr/bin/env python3

import sys
import serial
import rospy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fsr_main.msg import FSRValues

# Add the path to the custom module
sys.path.append(r"/home/asl1/catkin_ws/src/UR5/fsr_main")

# Initialize ROS node
rospy.init_node('fsr_recorder')

# Create a publisher for the FSRValues message type
fsr_values_pub = rospy.Publisher('/fsr_values', FSRValues, queue_size=10)

# Function to fit a polynomial and plot the results
def fit_polynomial(x, y, degree=7):
    """
    Fit a polynomial to the provided data and plot the results.

    Args:
        x (array-like): X values of the data.
        y (array-like): Y values of the data.
        degree (int): Degree of the polynomial to fit.

    Returns:
        polynomial_func (np.poly1d): Polynomial function that fits the data.
    """
    # Fit polynomial
    coefficients = np.polyfit(x, y, degree)

    # Create a polynomial function
    polynomial_func = np.poly1d(coefficients)

    # Generate x values for plotting
    x_plot = np.linspace(min(x), max(x), 100)

    # Calculate corresponding y values using the polynomial function
    y_plot = polynomial_func(x_plot)

    # Plot the original data and the fitted polynomial
    # plt.scatter(x, y, label='Original Data')
    # plt.plot(x_plot, y_plot, label=f'Polynomial Fit (Degree {degree})', color='red')
    # plt.xlabel('X Values')
    # plt.ylabel('Y Values')
    # plt.legend()
    # plt.show()

    return polynomial_func

# Read data from Excel sheets
def read_excel_and_fit(file_path, sheet_name, x_column, y_column):
    """
    Read data from an Excel sheet and fit a polynomial.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet in the Excel file.
        x_column (str): Name of the column containing X values.
        y_column (str): Name of the column containing Y values.

    Returns:
        polynomial_func (np.poly1d): Polynomial function that fits the data.
    """
    # Read Excel sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract X and Y values from DataFrame
    x_values = df[x_column].values
    y_values = df[y_column].values

    # Fit polynomial and plot results
    polynomial_func = fit_polynomial(x_values, y_values)

    return polynomial_func

# Set up the serial connection
serial_port = '/dev/ttyUSB0'
baud_rate = 921600
ser = serial.Serial(serial_port, baud_rate, timeout=.1)

# File paths to the Excel sheets
file_path2 = '/home/asl1/catkin_ws/src/UR5/fsr_main/src/sheets/FSR_2.xlsx'
file_path3 = '/home/asl1/catkin_ws/src/UR5/fsr_main/src/sheets/FSR_3.xlsx'
file_path4 = '/home/asl1/catkin_ws/src/UR5/fsr_main/src/sheets/FSR_4.xlsx'

# Excel sheet and column names
sheet_name = 'graph'  # Adjust as needed
x_column = 'FSR Output'  # Replace with the actual column name containing X values
y_column = 'Force Recorded'  # Replace with the actual column name containing Y values

# Fit polynomials for each Excel sheet
polynomial_func2 = read_excel_and_fit(file_path2, sheet_name, x_column, y_column)
polynomial_func3 = read_excel_and_fit(file_path3, sheet_name, x_column, y_column)
polynomial_func4 = read_excel_and_fit(file_path4, sheet_name, x_column, y_column)

# Initialize lists to store incoming FSR values
fsr_v_values = []
fsr_w_values = []
fsr_x_values = []

try:
    while not rospy.is_shutdown():
        # Read a line from the serial port
        line = ser.readline().strip().decode()

        # Check if the line starts with '<' and ends with '>'
        if line.startswith('<') and line.endswith('>'):
            # Remove '<' and '>'
            line = line[1:-1]

            # Split the values using ',' as the delimiter
            values = line.split(',')

            # Ensure there are three values
            if len(values) == 3:
                # Store the values in variables
                fsr_v, fsr_w, fsr_x = map(int, values)

                # Print or use the values as needed
                print(f"fsr_v: {fsr_v}, fsr_w: {fsr_w}, fsr_x: {fsr_x}")

                # Append values to the lists
                fsr_v_values.append(fsr_v)
                fsr_w_values.append(fsr_w)
                fsr_x_values.append(fsr_x)

                # Map the FSR values to force using the polynomial functions
                force_v = polynomial_func2(fsr_v_values[-1])
                force_w = polynomial_func3(fsr_w_values[-1])
                force_x = polynomial_func4(fsr_x_values[-1])

                # Display the mapped values
                print("Mapped Y Values for Input X Values:")
                print(f"Output Y Values (Sheet 2): {force_v}")
                print(f"Output Y Values (Sheet 3): {force_w}")
                print(f"Output Y Values (Sheet 4): {force_x}")

                # Create an instance of the custom message type
                fsr_values_msg = FSRValues()
                fsr_values_msg.fsr_v = fsr_v
                fsr_values_msg.fsr_w = fsr_w
                fsr_values_msg.fsr_x = fsr_x
                fsr_values_msg.force_v = force_v
                fsr_values_msg.force_w = force_w
                fsr_values_msg.force_x = force_x

                # Publish the custom message
                fsr_values_pub.publish(fsr_values_msg)

except KeyboardInterrupt:
    # Close the serial port on keyboard interrupt
    ser.close()
    print("Serial port closed.")
    rospy.signal_shutdown("KeyboardInterrupt")
