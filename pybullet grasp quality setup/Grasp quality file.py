# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:47:58 2024

@author: SHREYASH
"""
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

# Connect to PyBullet simulation
p.connect(p.GUI)

# Set the path to the URDF file
p.setAdditionalSearchPath(pybullet_data.getDataPath())

start_position = [0.0704, 0.0555, 0.2104]
orientation = [-1.5277,  0.0401, -0.5698]

# Convert Euler angles to rotation matrix
rotation = Rotation.from_euler('xyz', orientation)
rotation_matrix = rotation.as_matrix()

# The direction vector is the second column of the rotation matrix
direction = rotation_matrix[:, 1]

# Normalize the direction vector
direction_normalized = direction / np.linalg.norm(direction)

# Calculate the new point 5 cm away
distance = 0.02  # 5 cm in meters
new_point = start_position + direction_normalized * distance
# Load the plane and the Robotiq 3-finger gripper URDF
planeId = p.loadURDF("plane.urdf")

start_orientation = p.getQuaternionFromEuler(orientation)
robotId = p.loadURDF("C:\\Users\\SHREYASH\\Downloads\\robotiq-hydro-devel\\robotiq-hydro-devel\\robotiq_s_model_visualization\\cfg\\s-model_articulated.urdf", new_point, start_orientation)
objID = p.loadURDF("C:\\Users\\SHREYASH\\Downloads\\120 size objects\\120 size objects\\egad_eval_set_0.12\\processed_meshes\\obj_train.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)                 

p.setGravity(0, 0, 0)

# Set physics engine parameters
p.setPhysicsEngineParameter(numSolverIterations=150)
p.changeDynamics(objID, -1, lateralFriction=0.5, mass=0.0)
joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

p.changeDynamics(robotId, -1, lateralFriction=0.5, mass=0.0)
for joint_index in joint_indices:
    p.changeDynamics(robotId, joint_index, lateralFriction=0.5)

# Add sliders for joint control
slider_ids = []
for joint in joint_indices:
    if joint in [0, 4]:  # lateral_finger_1 and lateral_finger_2
        slider_ids.append(p.addUserDebugParameter(f'Joint {joint}', -0.25, 0.25, 0))
    else:
        slider_ids.append(p.addUserDebugParameter(f'Joint {joint}', -0.5, 1.5, 0))

# Add sliders for gripper position and orientation control
x_slider = p.addUserDebugParameter('X Position', -1, 1, new_point[0])
y_slider = p.addUserDebugParameter('Y Position', -1, 1, new_point[1])
z_slider = p.addUserDebugParameter('Z Position', 0, 1, new_point[2])
roll_slider = p.addUserDebugParameter('Roll', -math.pi, math.pi, orientation[0])
pitch_slider = p.addUserDebugParameter('Pitch', -math.pi, math.pi, orientation[1])
yaw_slider = p.addUserDebugParameter('Yaw', -math.pi, math.pi, orientation[2])

# Function to control gripper joints based on slider values
def update_gripper(robotId, slider_ids, joint_indices):
    target_positions = [p.readUserDebugParameter(slider_id) for slider_id in slider_ids]
    p.setJointMotorControlArray(robotId, joint_indices, p.POSITION_CONTROL, targetPositions=target_positions)

close_gripper_button = p.addUserDebugParameter('Close Gripper', 1, 0, 0)
open_gripper_button = p.addUserDebugParameter('Open Gripper', 1, 0, 0)


def open_gripper(robotId):
    for joint in [1, 5, 9]:
        p.setJointMotorControl2(robotId, joint, p.POSITION_CONTROL, targetPosition=0, force=100)
        
def calculate_epsilon_quality(contact_points, friction_coeff=0.5, force_threshold=0.01):
    G = []
    
    # Get the base position and orientation of the robot arm
    base_pos, base_orient = p.getBasePositionAndOrientation(robotId)
    
    for point in contact_points:
        # Transform contact point to world frame
        link_index = point[3]
        link_state = p.getLinkState(robotId, link_index)
        link_position = link_state[0]
        link_orientation = link_state[1]
        contact_position_local = point[5]
        contact_position_world = p.multiplyTransforms(link_position, link_orientation, contact_position_local, [0, 0, 0, 1])[0]
        
        # Transform world position to base frame
        contact_position_base = p.multiplyTransforms(base_pos, base_orient, contact_position_world, [0, 0, 0, 1])[0]
        
        position = np.array(contact_position_base)
        print('position:',position)
        normal = np.array(point[7])
        # Transform normal to base frame
        normal_world = p.rotateVector(link_orientation, normal)
        normal_base = p.rotateVector(p.invertTransform(base_pos, base_orient)[1], normal_world)
        normal = np.array(normal_base)
        
        normal_force = 100
        
        # Only consider contacts with normal force above the threshold
        if normal_force > force_threshold:
            # Compute the friction cone edges
            t1 = np.array([-normal[1], normal[0], 0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            
            for i in range(4):  # 4-sided friction cone approximation
                angle = i * math.pi / 2
                direction = math.cos(angle) * t1 + math.sin(angle) * t2
                force = normal * normal_force + friction_coeff * normal_force * direction
                force /= np.linalg.norm(force)
                
                wrench = np.concatenate([force, np.cross(position, force)])
                G.append(wrench)
    
    if not G:
        print("No valid contacts with sufficient normal force.")
        return 0
    
    G = np.array(G)
    
    # Compute the minimum singular value of G
    _, s, _ = np.linalg.svd(G)
    epsilon = s[-1]
    return epsilon

def visualize_contact_points(contact_points):
    for point in contact_points:
        start = point[5]
        end = [start[0] + point[7][0]*0.1,
               start[1] + point[7][1]*0.1,
               start[2] + point[7][2]*0.1]
        p.addUserDebugLine(start, end, lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=0)
        
        # Add a small line to represent the point itself
        point_end = [start[0] + 0.005, start[1] + 0.005, start[2] + 0.005]
        p.addUserDebugLine(start, point_end, lineColorRGB=[1, 0, 0], lineWidth=6, lifeTime=0)

def visualize_grasp(contact_points):
    for i in range(len(contact_points)):
        for j in range(i+1, len(contact_points)):
            p.addUserDebugLine(contact_points[i][5], contact_points[j][5],
                               lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)

def close_gripper(robotId):
    finger_joints = [1, 5, 9]  # Main joints for each finger
    max_angle = 1.5
    step = 0.05
    contact_detected = [False, False, False]
    all_contacts = []

    while not all(contact_detected):
        for i, joint in enumerate(finger_joints):
            if not contact_detected[i]:
                current_angle = p.getJointState(robotId, joint)[0]
                next_angle = min(current_angle + step, max_angle)
                p.setJointMotorControl2(robotId, joint, p.POSITION_CONTROL, targetPosition=next_angle, force=10000)

        p.stepSimulation()

        contacts = p.getContactPoints(robotId, objID)
        for contact in contacts:
            link_index = contact[3]
            if link_index in [2, 3, 6, 7, 10, 11]:  # Check if contact is on finger links
                finger_index = finger_joints.index(link_index - 1 if link_index % 2 == 0 else link_index - 2)
                if not contact_detected[finger_index]:
                    print(f"Contact detected for finger {finger_index + 1}:")
                    print(f"Link index: {link_index}, Position on link: {contact[5]}, Normal force: {contact[9]}, lateral_friction_1: {contact[10]}, lateral_friction_2: {contact[12]}")
                    contact_detected[finger_index] = True
                    current_pos = p.getJointState(robotId, finger_joints[finger_index])[0]
                    p.setJointMotorControl2(robotId, finger_joints[finger_index], p.POSITION_CONTROL, targetPosition=current_pos, force=10000)
                    all_contacts.append(contact)

        if all(contact_detected) or all(p.getJointState(robotId, joint)[0] >= max_angle for joint in finger_joints):
            break

    # Make object and gripper static after grasping
    p.changeDynamics(robotId, -1, mass=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(objID, -1, mass=0, linearDamping=0, angularDamping=0)

    # Fix the object in place
    p.createConstraint(robotId, -1, objID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

    # Calculate and display epsilon quality
    if all_contacts:
        epsilon_quality = calculate_epsilon_quality(all_contacts)
        print(f"Epsilon Quality: {epsilon_quality}")

        # Visualize contact points and grasp
        visualize_contact_points(all_contacts)
        visualize_grasp(all_contacts)
    else:
        print("No contacts detected. Unable to calculate epsilon quality.")

    return True  # Return True to indicate the gripper has closed

gripper_closed = False

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    
    if not gripper_closed:
        update_gripper(robotId, slider_ids, joint_indices)
        
        x_pos = p.readUserDebugParameter(x_slider)
        y_pos = p.readUserDebugParameter(y_slider)
        z_pos = p.readUserDebugParameter(z_slider)
        roll = p.readUserDebugParameter(roll_slider)
        pitch = p.readUserDebugParameter(pitch_slider)
        yaw = p.readUserDebugParameter(yaw_slider)
        
        new_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
        p.resetBasePositionAndOrientation(robotId, [x_pos, y_pos, z_pos], new_orientation)
    
    if p.readUserDebugParameter(close_gripper_button) > 0.5 and not gripper_closed:
        gripper_closed = close_gripper(robotId)
    
    if p.readUserDebugParameter(open_gripper_button) > 0.5 and gripper_closed:
        # Restore dynamics before opening
        p.changeDynamics(robotId, -1, mass=1, linearDamping=0.04, angularDamping=0.04)
        p.changeDynamics(objID, -1, mass=1, linearDamping=0.04, angularDamping=0.04)
        open_gripper(robotId)
        # Remove the fixed constraint
        for i in range(p.getNumConstraints()):
            p.removeConstraint(i)
        # Reset the open gripper button
        p.resetDebugVisualizerItem(open_gripper_button)
        gripper_closed = False

