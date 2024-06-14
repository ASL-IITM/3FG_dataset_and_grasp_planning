# Force Control and Slip Detection for a Non-Backdrivable Robotic Gripper

## What the Project Does

This project focuses on the development and implementation of a force control and slip detection system for a non-backdrivable robotic gripper. The system aims to enhance the gripper's ability to handle objects with varying levels of grip strength, ensuring that it can securely hold items without causing damage. The project includes the following components:
- Force control algorithms to adjust the gripping force based on real-time feedback.
- Slip detection mechanisms to identify when an object is slipping and adjust the grip accordingly.
- Experimental validation and testing of the system using various objects and scenarios.

### Force Control Demonstration

![Force Control Video](https://force-control-video)

### Slip Detection Demonstration

![Slip Detection Video](https://slip-detection-video)

## Why the Project is Useful

This project is particularly useful for applications in industrial automation, where precise handling of delicate or variable objects is critical. The ability to control grip force and detect slip in real-time can:
- Reduce damage to sensitive components.
- Improve the efficiency and reliability of automated handling systems.
- Enable the gripper to adapt to a wide range of objects, increasing its versatility.

## How Users Can Get Started with the Project

To get started with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/JosephJoshyK/Force-Control-and-Slip-Detection-for-a-Non-Backdrivable-Robotic-Gripper.git
    cd Force-Control-and-Slip-Detection-for-a-Non-Backdrivable-Robotic-Gripper
    ```

3. **Setup the Hardware:** Follow the instructions in the `FSR Setup`, `Gripper Setup`, and `Complete Setup` folders to set up the robotic gripper and connect the FSR sensors.

    - **FSR Setup:**
      - Navigate to the `FSR Setup` folder.
      - Follow the detailed setup guide to correctly position and connect the FSR sensors.

    - **Gripper Setup:**
      - Navigate to the `Gripper Setup` folder.
      - Follow the step-by-step instructions to configure the robotic gripper.

    - **Complete Setup:**
      - Navigate to the `Complete Setup` folder.
      - Follow the comprehensive guide to ensure all components are correctly integrated and configured.


4. **Launch the fsr_main codes:**
    ```bash
    roslaunch fsr_main smart_grip.launch
    ```

For more detailed instructions, refer to the `README.md` in each folder.

## Where Users Can Get Help with Your Project

If you encounter any issues or have questions about the project, you can:
- **Open an Issue:** Create a new issue in the GitHub repository for any bugs or feature requests.
- **Discussions:** Participate in discussions on the GitHub Discussions page for this project.
- **Documentation:** Refer to the `README.md` folder for detailed documentation on the system architecture, setup, and usage.

## Who Maintains and Contributes to the Project

This project is maintained by Joseph Joshy. Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes. Ensure your code adheres to the project's coding standards and includes relevant tests.

---

Feel free to reach out if you have any questions or need further assistance. Thank you for your interest in our project!

