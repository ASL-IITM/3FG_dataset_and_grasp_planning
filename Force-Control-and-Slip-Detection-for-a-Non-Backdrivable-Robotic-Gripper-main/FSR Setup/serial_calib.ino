#define FSR_PIN 2  // Pin connected to the FSR

int fsrValue;  // Variable to store the FSR reading

void setup() {
    // Initialize the FSR pin as an input
    pinMode(FSR_PIN, INPUT);
    
    // Start the serial communication at 115200 baud rate
    Serial.begin(115200);
    
    // Small delay to ensure everything is set up properly
    delay(500);
}

void loop() {
    // Read the value from the FSR
    fsrValue = analogRead(FSR_PIN);
    
    // Send the FSR value over serial communication
    Serial.print("<");
    Serial.print(fsrValue);
    Serial.println(">");
    
    // Small delay to limit the number of readings per second
    // delay(100); // Optional: Adjust the delay as needed
}

