# FSR Setup

## Overview

This folder contains the necessary files and instructions to set up the Force Sensitive Resistor (FSR) sensors for the robotic gripper. The setup includes uploading code to an ESP32 microcontroller, connecting the FSR to the ADC port, and calibrating the sensor using data collected from the ESP32 and ATI-Mini loadcell.

## Contents

- `serial_calib.ino`: Arduino code to upload to the ESP32 for reading FSR data.
- `ati_mini.cpp`: Code to interface with the ATI-Mini and collect data in parallel with the ESP32.
- `FSR_A.xlsx`,`FSR_B.xlsx`,`FSR_C.xlsx`: Excel file used to store and process the calibration data.

## Hardware Setup

1. **Connect the FSR to ESP32:**
    - Connect the FSR sensor to the 2nd ADC port on the ESP32.

2. **Upload the `serial_calib.ino` Code:**
    - Open the `serial_calib.ino` file in the Arduino IDE.
    - Select the correct board (ESP32) and port.
    - Upload the code to the ESP32.

## Calibration Process

1. **Ensure ESP32 serial communication**
    - Ensure the ESP32 is connected to your system via a serial connection.
    - The ESP32 will start reading data from the FSR and send it over serial communication.

2. **Run the `ati_mini.cpp` Code:**
    - Ensure the ATI mini loadcell is connected to your system via ethernet cable.
    - Compile and run the `ati_mini.cpp` file on your system.
    - This code will interface with the ATI-mini and sync the collected data in parallel with the ESP32 data.

4. **Collect Data:**
    - The data from both the ESP32 and the ATI-mini will be collected and stored in an Excel file.
    - Each sheet will have:
        - Column 5: Data from the ATI-mini which is force applied.
        - Column 9: Data from the ESP32.
    - Excel file for each force value is uploaded as different sheets in single Excel file as shown in `FSR_A.xlsx`,`FSR_B.xlsx`,`FSR_C.xlsx`.
    - Use the data in the `xlsx` file to derive the calibration equations for the FSR.
    - The final sheet in the Excel file will contain the averaged values and the curve.

## Calibration Equations

After processing the data, you will obtain calibration equations for the FSR. These equations will be used in the main application to convert raw sensor readings into meaningful force measurements.

Example calibration equation:
Force = a * {FSR Reading}^3 + b * {FSR Reading}^2 + c * {FSR Reading} + d 

## Example Calibration Equations

Example:
```markdown
## Calibration Equations

Based on the calibration data from FSR_A.xlsx, FSR_B.xlsx and FSR_C.xlsx for three different FSRs recpectively,
the following equations were derived for the FSR:

FA = 2.61e−9x^3 − 4.50e−6x^2 + 0.009x + 0.167
FB = 1.22e−9x^3 − 7.99e−7x^2 + 0.007x + 1.162
FC = 3.82e−9x^3 − 7.86e−6x^2 + 0.012x − 0.225
where x denotes the FSR Reading

These equations are used in the main application to convert the raw FSR readings into force measurements.

