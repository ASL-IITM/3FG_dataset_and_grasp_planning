# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:32:13 2024

@author: Shreyash Gadgil
"""
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.transform import Rotation as R

basic_grasp_vertices = np.array([[0.09, 0, 0], [-0.09, 0, 0.04], [-0.09, 0, -0.04]])

def quaternion_loss(q_actual, q_predicted):
    """
    Compute quaternion loss between actual and predicted quaternions.

    Args:
        q_actual (torch.Tensor): Actual quaternions (batch_size x num_outputs x 4).
        q_predicted (torch.Tensor): Predicted quaternions (batch_size x num_outputs x 4).

    Returns:
        torch.Tensor: Quaternion loss.
    """
    # Compute dot product
    #dot_product = torch.sum(q_actual * q_predicted, dim=2)

    # Take absolute difference from 1
    #loss = 1 - torch.abs(dot_product)
    #loss = torch.acos(dot_product)
    loss = torch.abs(torch.acos(torch.clamp(torch.sum(q_actual * q_predicted, dim=2), -1.0, 1.0)))
    #print('loss:',loss.mean())

    return loss.mean()  # Return mean loss over the batch and number of outputs

def euler_to_unit_vector(angles):
    """
    Convert Euler angles to unit vector representation.

    Args:
        angles (torch.Tensor): Euler angles (batch_size x num_outputs x 3).

    Returns:
        torch.Tensor: Unit vectors (batch_size x num_outputs x 3).
    """
    roll, pitch, yaw = angles.unbind(dim=-1)

    # Compute unit vectors
    x = torch.cos(pitch) * torch.cos(yaw)
    y = torch.sin(roll) * torch.sin(pitch) * torch.cos(yaw) - torch.cos(roll)*torch.sin(yaw)
    z = torch.cos(roll) * torch.sin(pitch) * torch.cos(yaw) + torch.sin(roll)*torch.sin(yaw)

    unit_vector = torch.stack((x, y, z), dim=-1)

    return unit_vector


def angular_difference_loss(angles_actual, angles_predicted):
    """
    Compute angular difference loss between actual and predicted Euler angles.

    Args:
        angles_actual (torch.Tensor): Actual Euler angles (batch_size x num_outputs x 3).
        angles_predicted (torch.Tensor): Predicted Euler angles (batch_size x num_outputs x 3).

    Returns:
        torch.Tensor: Angular difference loss.
    """
    # Convert Euler angles to unit vectors
    actual_unit_vector = euler_to_unit_vector(angles_actual)
    predicted_unit_vector = euler_to_unit_vector(angles_predicted)

    # Compute dot product between actual and predicted unit vectors
    dot_product = torch.sum(actual_unit_vector * predicted_unit_vector, dim=-1)

    # Clamp dot product to ensure it falls within valid range [-1, 1] for acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute angular difference
    angle_difference = torch.acos(dot_product)

    # Return mean angular difference over the batch and number of outputs
    return angle_difference.mean()


def accuracy(pos_predictions, pos_targets, orient_predictions, orient_targets, pos_threshold=0.1, ori_threshold=0.5236):
    """
    Calculate the accuracy based on Euclidean distance between predictions and targets.

    Args:
    - predictions (torch.Tensor): Predicted positions (batch_size, num_outputs, 3).
    - targets (torch.Tensor): Actual positions (batch_size, num_outputs, 3).
    - threshold (float): Maximum allowed Euclidean distance for a prediction to be considered accurate.

    Returns:
    - accuracy (float): Percentage of accurate predictions.
    """
    batch_size, num_outputs, _ = pos_predictions.size()
    num_correct = 0

    for i in range(batch_size):
        for j in range(num_outputs):
            pos_pred = pos_predictions[i, j]
            pos_target = pos_targets[i, j]
            #ori_pred = orient_predictions[i, j]
            #ori_target = orient_targets[i, j]
            distance = torch.norm(pos_pred - pos_target)  # Calculate Euclidean distance
            ori_pred = euler_to_unit_vector(orient_predictions[i, j])  # Convert Euler angles to unit vectors
            ori_target = euler_to_unit_vector(orient_targets[i, j])  # Convert Euler angles to unit vectors
            angle = torch.abs(torch.acos(torch.clamp(torch.sum(ori_target * ori_pred, dim=-1), -1.0, 1.0)))  # Calculate angle between unit vectors
            if distance <= pos_threshold and angle <= ori_threshold:
            #if distance <= pos_threshold:
                num_correct += 1
    total_predictions = batch_size * num_outputs
    accuracy = (num_correct / total_predictions) * 100.0
    #print('accuracy: ',accuracy)

    return accuracy

def regression_metrics(pos_1_predictions, pos_1_targets, pos_2_predictions, pos_2_targets, pos_3_predictions, pos_3_targets):
    """
    Calculate regression metrics for position and orientation predictions.

    Args:
    - pos_predictions (torch.Tensor): Predicted positions (batch_size, num_outputs, 3).
    - pos_targets (torch.Tensor): Actual positions (batch_size, num_outputs, 3).
    - orient_predictions (torch.Tensor): Predicted orientations (batch_size, num_outputs, 3).
    - orient_targets (torch.Tensor): Actual orientations (batch_size, num_outputs, 3).

    Returns:
    - pos_rmse (float): Root Mean Squared Error for position predictions.
    - pos_r2 (float): R-squared for position predictions.
    - orient_mae (float): Mean Absolute Error for orientation predictions.
    - orient_r2 (float): R-squared for orientation predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Position metrics
    pos_1_rmse = torch.sqrt(torch.mean(torch.square(pos_1_predictions - pos_1_targets)))
    pos_1_r2 = r2_score(pos_1_targets.cpu().numpy().reshape(-1, 3), pos_1_predictions.cpu().numpy().reshape(-1, 3))
    
    # Position metrics
    pos_2_rmse = torch.sqrt(torch.mean(torch.square(pos_2_predictions - pos_2_targets)))
    pos_2_r2 = r2_score(pos_2_targets.cpu().numpy().reshape(-1, 3), pos_2_predictions.cpu().numpy().reshape(-1, 3))
    
    # Position metrics
    pos_3_rmse = torch.sqrt(torch.mean(torch.square(pos_3_predictions - pos_3_targets)))
    pos_3_r2 = r2_score(pos_3_targets.cpu().numpy().reshape(-1, 3), pos_3_predictions.cpu().numpy().reshape(-1, 3))
    
    
    return pos_1_rmse.item(), pos_1_r2, pos_2_rmse.item(), pos_2_r2, pos_3_rmse.item(), pos_3_r2



def compute_euler_angles(basic_grasp_vertices, new_vertices):
    """
    Compute Euler angles (roll, pitch, yaw) between original and new vertices.
    Args:
        basic_grasp_vertices (torch.Tensor): Fixed original vertices (3, 3).
        new_vertices (torch.Tensor): Predicted vertices (batch_size, num_outputs, 3).
    Returns:
        euler_angles (torch.Tensor): Computed Euler angles (batch_size, 3).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure basic_grasp_vertices is a torch tensor
    if not isinstance(basic_grasp_vertices, torch.Tensor):
        basic_grasp_vertices = torch.tensor(basic_grasp_vertices, dtype=torch.float32)
    basic_grasp_vertices = basic_grasp_vertices.to(device)
    
    # Reshape new_vertices to (batch_size, 3, 3)
    new_vertices = new_vertices.view(new_vertices.shape[0], 3, 3).to(device)

    # Compute the centroid for the basic grasp vertices
    centroid_original = basic_grasp_vertices.mean(dim=0)

    # Compute the centroids for the new vertices
    centroid_new = new_vertices.mean(dim=1)

    # Center the vertices
    original_centered = basic_grasp_vertices - centroid_original
    new_centered = new_vertices - centroid_new[:, None]

    # Initialize list to store Euler angles
    euler_angles_list = []

    # Iterate over the batch to compute Euler angles
    for i in range(new_centered.shape[0]):
        H = torch.matmul(original_centered.T, new_centered[i])
        U, S, Vt = torch.svd(H)
        rotation_matrix = torch.matmul(Vt, U.T)

        # Ensure a proper right-handed coordinate system
        if torch.det(rotation_matrix) < 0:
            Vt[:, 2] *= -1
            rotation_matrix = torch.matmul(Vt, U.T)

        # Convert the rotation matrix to Euler angles (roll, pitch, yaw)
        r = R.from_matrix(rotation_matrix.cpu().numpy())
        euler_angles = r.as_euler('xyz', degrees=True)  # 'xyz' order corresponds to roll, pitch, yaw
        euler_angles_list.append(np.radians(euler_angles))

    # Convert the list of Euler angles to a tensor
    euler_angles_array = np.array(euler_angles_list)
    euler_angles_tensor = torch.tensor(euler_angles_array, dtype=torch.float32).to(device)
    #euler_angles_tensor = torch.tensor(euler_angles_list, dtype=torch.float32)

    return euler_angles_tensor


def calculate_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode
    total_batch_pos_1_rmse = 0.0
    total_batch_pos_1_r2 = 0.0
    total_batch_pos_2_rmse = 0.0
    total_batch_pos_2_r2 = 0.0
    total_batch_pos_3_rmse = 0.0
    total_batch_pos_3_r2 = 0.0

    

    with torch.no_grad():
        for rgbd_images, positions_1, positions_2, positions_3, centroid, orientation, d in dataloader:
            rgbd_images = rgbd_images.to(device)
            positions_1 = positions_1.to(device)
            positions_2 = positions_2.to(device)
            positions_3 = positions_3.to(device)
            
            positions_1 = positions_1.float()
            positions_2 = positions_2.float()
            positions_3 = positions_3.float()
            
            pos_1_outputs, pos_2_outputs, pos_3_outputs, _, _, _ = model(rgbd_images)
            batch_pos_1_rmse, batch_pos_1_r2, batch_pos_2_rmse, batch_pos_2_r2, batch_pos_3_rmse, batch_pos_3_r2 = regression_metrics(pos_1_predictions = pos_1_outputs, pos_1_targets = positions_1, pos_2_predictions = pos_2_outputs, pos_2_targets = positions_2 , pos_3_predictions = pos_3_outputs, pos_3_targets = positions_3)
            #batch_accuracy = accuracy(pos_predictions = pos_outputs, pos_targets = positions, orient_predictions=orient_outputs, orient_targets=orientations )  # Use the previously defined accuracy function
            total_batch_pos_1_rmse += batch_pos_1_rmse
            total_batch_pos_1_r2 += batch_pos_1_r2
            total_batch_pos_2_rmse += batch_pos_2_rmse
            total_batch_pos_2_r2 += batch_pos_2_r2
            total_batch_pos_3_rmse += batch_pos_3_rmse
            total_batch_pos_3_r2 += batch_pos_3_r2


    model.train()  # Set the model back to training mode
    return total_batch_pos_1_rmse / len(dataloader), total_batch_pos_1_r2/len(dataloader), total_batch_pos_2_rmse / len(dataloader), total_batch_pos_2_r2/len(dataloader), total_batch_pos_3_rmse / len(dataloader), total_batch_pos_3_r2/len(dataloader)# Return average accuracy

# =============================================================================
# def calculate_accuracy(model, dataloader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()  # Set the model to evaluation mode
#     total_batch_pos_1_rmse = 0.0
#     total_batch_pos_1_r2 = 0.0
#     total_batch_pos_2_rmse = 0.0
#     total_batch_pos_2_r2 = 0.0
#     total_batch_pos_3_rmse = 0.0
#     total_batch_pos_3_r2 = 0.0
#     total_batch_centroid_rmse = 0.0
#     total_batch_centroid_r2 = 0.0
#     total_batch_euler_angle = 0.0
#     total_batch_d_rmse = 0.0
#     total_batch_d_r2 = 0.0
#     
# 
#     with torch.no_grad():
#         for rgbd_images, positions_1, positions_2, positions_3, centroid, orientation, d in dataloader:
#             rgbd_images = rgbd_images.to(device)
#             positions_1 = positions_1.to(device)
#             positions_2 = positions_2.to(device)
#             positions_3 = positions_3.to(device)
#             centroid = centroid.to(device)
#             orientation = orientation.to(device)
#             d = d.to(device)
#             
#             positions_1 = positions_1.float()
#             positions_2 = positions_2.float()
#             positions_3 = positions_3.float()
#             centroid = centroid.float()
#             orientation = orientation.float()
#             d = d.float()
#             
#             pos_1_outputs, pos_2_outputs, pos_3_outputs, d_output = model(rgbd_images)
#             predicted_centroid = (pos_1_outputs + pos_2_outputs + pos_3_outputs)/3
#             #pos_outputs = model(rgbd_images)
#             batch_pos_1_rmse, batch_pos_1_r2, batch_pos_2_rmse, batch_pos_2_r2, batch_pos_3_rmse, batch_pos_3_r2, batch_centroid_rmse, batch_centroid_r2, euler_angle_difference, batch_d_rmse, batch_d_r2 = regression_metrics(pos_1_predictions = pos_1_outputs, pos_1_targets = positions_1, pos_2_predictions = pos_2_outputs, pos_2_targets = positions_2 , pos_3_predictions = pos_3_outputs, pos_3_targets = positions_3, centroid = centroid, predicted_centroid = predicted_centroid, orientation = orientation, expected_d = d, predicted_d = d_output)
#             #batch_accuracy = accuracy(pos_predictions = pos_outputs, pos_targets = positions, orient_predictions=orient_outputs, orient_targets=orientations )  # Use the previously defined accuracy function
#             total_batch_pos_1_rmse += batch_pos_1_rmse
#             total_batch_pos_1_r2 += batch_pos_1_r2
#             total_batch_pos_2_rmse += batch_pos_2_rmse
#             total_batch_pos_2_r2 += batch_pos_2_r2
#             total_batch_pos_3_rmse += batch_pos_3_rmse
#             total_batch_pos_3_r2 += batch_pos_3_r2
#             total_batch_centroid_rmse += batch_centroid_rmse
#             total_batch_centroid_r2 += batch_centroid_r2
#             total_batch_euler_angle += euler_angle_difference
#             total_batch_d_rmse += batch_d_rmse
#             total_batch_d_r2 += batch_d_r2
# 
#     model.train()  # Set the model back to training mode
#     return total_batch_pos_1_rmse / len(dataloader), total_batch_pos_1_r2/len(dataloader), total_batch_pos_2_rmse / len(dataloader), total_batch_pos_2_r2/len(dataloader), total_batch_pos_3_rmse / len(dataloader), total_batch_pos_3_r2/len(dataloader), total_batch_centroid_rmse/len(dataloader), total_batch_centroid_r2/len(dataloader), total_batch_euler_angle/len(dataloader), total_batch_d_rmse/len(dataloader), total_batch_d_r2/len(dataloader)    # Return average accuracy
# 
# =============================================================================
# =============================================================================
# def regression_metrics(pos_1_predictions, pos_1_targets, pos_2_predictions, pos_2_targets, pos_3_predictions, pos_3_targets, centroid, predicted_centroid, orientation, predicted_d, expected_d):
#     """
#     Calculate regression metrics for position and orientation predictions.
# 
#     Args:
#     - pos_predictions (torch.Tensor): Predicted positions (batch_size, num_outputs, 3).
#     - pos_targets (torch.Tensor): Actual positions (batch_size, num_outputs, 3).
#     - orient_predictions (torch.Tensor): Predicted orientations (batch_size, num_outputs, 3).
#     - orient_targets (torch.Tensor): Actual orientations (batch_size, num_outputs, 3).
# 
#     Returns:
#     - pos_rmse (float): Root Mean Squared Error for position predictions.
#     - pos_r2 (float): R-squared for position predictions.
#     - orient_mae (float): Mean Absolute Error for orientation predictions.
#     - orient_r2 (float): R-squared for orientation predictions.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Position metrics
#     pos_1_rmse = torch.sqrt(torch.mean(torch.square(pos_1_predictions - pos_1_targets)))
#     pos_1_r2 = r2_score(pos_1_targets.cpu().numpy().reshape(-1, 3), pos_1_predictions.cpu().numpy().reshape(-1, 3))
#     
#     # Position metrics
#     pos_2_rmse = torch.sqrt(torch.mean(torch.square(pos_2_predictions - pos_2_targets)))
#     pos_2_r2 = r2_score(pos_2_targets.cpu().numpy().reshape(-1, 3), pos_2_predictions.cpu().numpy().reshape(-1, 3))
#     
#     # Position metrics
#     pos_3_rmse = torch.sqrt(torch.mean(torch.square(pos_3_predictions - pos_3_targets)))
#     pos_3_r2 = r2_score(pos_3_targets.cpu().numpy().reshape(-1, 3), pos_3_predictions.cpu().numpy().reshape(-1, 3))
#     
#     centroid_rmse = torch.sqrt(torch.mean(torch.square(predicted_centroid - centroid)))
#     centroid_r2 = r2_score(centroid.cpu().numpy().reshape(-1, 3), predicted_centroid.cpu().numpy().reshape(-1, 3))
#     
#     # Compute new Euler angles from predicted positions
#     basic_grasp_vertices = torch.tensor([[0.09, 0, 0], [-0.09, 0, 0.04], [-0.09, 0, -0.04]], dtype=torch.float32)
#     new_vertices = torch.stack([pos_1_predictions, pos_2_predictions, pos_3_predictions], dim=1)
#     new_euler_angle = compute_euler_angles(basic_grasp_vertices, new_vertices).to(device)
# 
#     # Compute the mean absolute difference between predicted and actual Euler angles
#     euler_angle_difference = torch.mean(torch.abs(orientation - new_euler_angle))
#     
#     # Compute metrics for variable d
#     d_rmse = torch.sqrt(torch.mean(torch.square(predicted_d - expected_d)))
#     d_r2 = r2_score(expected_d.cpu().numpy().reshape(-1, 1), predicted_d.cpu().numpy().reshape(-1, 1))
#     
#     return pos_1_rmse.item(), pos_1_r2, pos_2_rmse.item(), pos_2_r2, pos_3_rmse.item(), pos_3_r2, centroid_rmse.item(), centroid_r2, euler_angle_difference, d_rmse, d_r2
# 
# 
# =============================================================================
