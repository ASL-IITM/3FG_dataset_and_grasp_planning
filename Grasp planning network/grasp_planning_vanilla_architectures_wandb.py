# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:32:13 2024

@author: Shreyash Gadgil
"""
#Torch dependencies
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Classes dependencies
from Dataloader import RGBDDataset
from Vanilla_CNN_models import CNN_Position_Orientation, ResNet_Position_Orientation
from Loss_and_accuracy import calculate_accuracy, quaternion_loss


'''# Example usage to check output of model:
# Instantiate the model
model = CNN_Position_Orientation(num_outputs=1)

# Assuming rgb_d_image is your input RGBD image tensor of shape (batch_size, channels, height, width)
# Forward pass

positions = model(a)
print(positions.shape)
# positions shape: (batch_size, num_outputs, 3)
# orientations shape: (batch_size, num_outputs, 4)
'''



if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
      print('Using GPU...!')
    else:
      print('Using CPU...!(terminate the runtime and restart using GPU)')
    #####################################################################################
    ############################  DATA LOADERS  #########################################
    #####################################################################################
    # Defining transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to match input size of CNN architecture if required
        #transforms.CenterCrop(512), #if we want to use the exact image without losing aspect ratio
        transforms.ToTensor(),  # Convert images to tensors
    ])

    # Initialize dataset
    rgb_dir = 'C:\\Users\\aslwo\\Downloads\\Dataset\\rgb'
    depth_dir = 'C:\\Users\\aslwo\\Downloads\\Dataset\\depth'
    csv_file = 'C:\\Users\\aslwo\\Downloads\\merged_top_1_entries.csv'
    dataset = RGBDDataset(rgb_dir,csv_file, depth_dir, transform=transform)

    # Initialize DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("DATASET SANITY CHECK OUTPUTS")
    print(f"Length of val dataset loader:{len(dataloader)}")
    a,b,c = (next(iter(dataloader)))
    print('IMAGE SHAPE:',a.shape)
    print('POSITION SHAPE:',b.shape)
    print('ORIENTAION SHAPE:',c.shape)
    print("-----------------------------------------------")
    # Define the training parameters
    epochs = 10
    learning_rate = 0.01

    # Initialize the model, loss function, and optimizer
    #model = CNN_Position_Orientation(num_outputs=1).to(device)
    model = ResNet_Position_Orientation(num_outputs=1).to(device)
    position_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        for batch_idx, (rgbd_images, positions, orientations) in enumerate(dataloader):
            # Forward pass
            rgbd_images = rgbd_images.to(device)
            positions = positions.to(device)
            orientations = orientations.to(device)

            positions= positions.float()
            orientations = orientations.float()
            pos_outputs, orient_outputs = model(rgbd_images)
             
            # Compute the loss
            loss_1 = position_criterion(pos_outputs, positions)
            #loss_2  = quaternion_loss(orientations, orient_outputs)
            loss_2  = position_criterion(orient_outputs,orientations)
            print('loss1, loss2',loss_1, loss_2)
            loss = loss_1*0.5 + loss_2*0.5

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print batch statistics
            total_loss += loss.item()
            if batch_idx % 10 == 9:  # Print every 10 batches
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, batch_idx+1, len(dataloader), total_loss / 10))
                total_loss = 0.0
            # Calculate accuracy after each epoch
        epoch_accuracy = calculate_accuracy(model, dataloader)
        print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch+1, epochs, epoch_accuracy))
        print("-----------------------------------------------")
        total_accuracy += epoch_accuracy

    print('Training finished!')
