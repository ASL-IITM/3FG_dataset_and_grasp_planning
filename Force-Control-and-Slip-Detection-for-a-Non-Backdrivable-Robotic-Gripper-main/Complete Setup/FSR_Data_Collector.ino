#include <Arduino.h>

// Define the pins for the Force Sensitive Resistors (FSRs)
const int fsrPin2 = 2;  // ADC pin for the first FSR
const int fsrPin3 = 33; // ADC pin for the second FSR
const int fsrPin4 = 32; // ADC pin for the third FSR

// Define the baud rate for serial communication
const int baudRate = 921600;

void setup() {
  // Initialize serial communication at the defined baud rate
  Serial.begin(baudRate);
}

void loop() {
  // Read the analog values from the FSRs
  int fsrValue2 = analogRead(fsrPin2);
  int fsrValue3 = analogRead(fsrPin3);
  int fsrValue4 = analogRead(fsrPin4);

  // Construct a data packet with start and end markers
  Serial.print('<'); // Start marker
  Serial.print(fsrValue2);
  Serial.print(",");
  Serial.print(fsrValue3);
  Serial.print(",");
  Serial.print(fsrValue4);
  Serial.print('>'); // End marker

  // Add a delay to control the data transmission rate
  delay(102); // Adjust the delay as needed
}

