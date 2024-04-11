# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:32:13 2024

@author: Shreyash Gadgil
"""
import torch

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
    dot_product = torch.sum(q_actual * q_predicted, dim=2)

    # Take absolute difference from 1
    loss = 1 - (torch.abs(dot_product))**2
    #loss = torch.acos(dot_product)
    #print('loss:',loss.mean())

    return loss.mean()  # Return mean loss over the batch and number of outputs



def accuracy(pos_predictions, pos_targets, orient_predictions, orient_targets, pos_threshold=0.05, ori_threshold=0.10471974):
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
            ori_pred = orient_predictions[i, j]
            ori_target = orient_targets[i, j]
            distance = torch.norm(pos_pred - pos_target)  # Calculate Euclidean distance
            #print('ori_pred: ', ori_pred)
            #print('ori_target: ', ori_target)
            angle = torch.acos(torch.sum(ori_target * ori_pred)) # 3 degree threshold for angle
            #print('angle: ', angle)
            #if distance <= pos_threshold and angle <= ori_threshold:
            if distance <= pos_threshold:
                num_correct += 1
    total_predictions = batch_size * num_outputs
    accuracy = (num_correct / total_predictions) * 100.0
    #print('accuracy: ',accuracy)

    return accuracy


def calculate_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode
    total_accuracy = 0.0

    with torch.no_grad():
        for rgbd_images, positions, orientations in dataloader:
            rgbd_images = rgbd_images.to(device)
            positions = positions.to(device)
            orientations = orientations.to(device)
            positions = positions.float()
            pos_outputs, orient_outputs = model(rgbd_images)
            batch_accuracy = accuracy(pos_predictions = pos_outputs, pos_targets = positions, orient_predictions = orient_outputs, orient_targets = orientations )  # Use the previously defined accuracy function
            total_accuracy += batch_accuracy

    model.train()  # Set the model back to training mode
    return total_accuracy / len(dataloader)  # Return average accuracy

