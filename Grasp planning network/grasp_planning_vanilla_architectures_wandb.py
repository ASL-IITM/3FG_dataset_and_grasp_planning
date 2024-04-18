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
#from torchsummary import summary
import wandb

#Classes dependencies
from Dataloader import RGBDDataset
from Vanilla_CNN_models import CNN_Position_Orientation, ResNet_Position_Orientation, EfficientNet_Position_Orientation
from Loss_and_accuracy import calculate_accuracy, quaternion_loss



if __name__ == '__main__':
    #####################################################################################
    ############################  GPU AVAILABILITY  #####################################
    #####################################################################################
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
      print('Using GPU...!')
    else:
      print('Using CPU...!(terminate the runtime and restart using GPU)')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
      
    ######################################################################################  
    sweep_config = {
        'method': 'bayes', #grid, random,bayes
        'metric': {
          'name': 'val_accuracy',
          'goal': 'maximize'  
        },
        'parameters': {
            'model': {
                'values': ['ResNet50','EfficientNet', 'ConvNet']
            },
            'learning_rate': {
                'values': [0.0001, 0.0002, 0.0003, 0.00025, 0.00001, 0.00002, 0.00003]
            },        
            'pretrained_weights':{
                'values':[True, False]
            },
            'loss_for_ori':{
                'values':['MSE', 'quaternion']
            },
            'loss_function_for_weights':{
                'values':[[0.5, 0.5], [0, 1], [1, 0], [0.7, 0.3]]
            },
            
        }
    }
        
    #####################################################################################
    ############################  DATA LOADERS  #########################################
    #####################################################################################
    # Defining transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match input size of CNN architecture if required
        #transforms.CenterCrop(512), #if we want to use the exact image without losing aspect ratio
        transforms.ToTensor()
        # Convert images to tensors
    ])

    # Initialize dataset
    rgb_dir = 'D:\\Grasp planning\dataset_1(depth=1,2)\\rgb'
    depth_dir = 'D:\\Grasp planning\\dataset_1(depth=1,2)\\depth'
    csv_file = 'D:\\Grasp planning\\merged_top_1_entries.csv'
    dataset = RGBDDataset(rgb_dir,csv_file, depth_dir, transform=transform)

    rgb_dir_eval = 'D:\\Grasp planning\\evaluation dataset\\rgb'
    depth_dir_eval = 'D:\\Grasp planning\\evaluation dataset\\depth'
    csv_file_eval = 'D:\\Grasp planning\\merged_top_1_entries_eval.csv'
    dataset_eval = RGBDDataset(rgb_dir_eval,csv_file_eval, depth_dir_eval, transform=transform)
    
    # Initialize DataLoader
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    '''
    print("DATASET SANITY CHECK OUTPUTS")
    print(f"Length of val dataset loader:{len(dataloader)}")
    a,b,c = (next(iter(dataloader)))
    print('IMAGE SHAPE:',a.shape)
    print('POSITION SHAPE:',b.shape)
    print('ORIENTAION SHAPE:',c.shape)
    print(f"Length of evaluation dataset loader:{len(dataloader_eval)}")
    a,b,c = (next(iter(dataloader_eval)))
    print('IMAGE SHAPE:',a.shape)
    print('POSITION SHAPE:',b.shape)
    print('ORIENTAION SHAPE:',c.shape)
    print("-----------------------------------------------")'''
    
    
    
    def sweep_train():
        config_defaults = {
            'Model':'ResNet50',
            'learning_rate':0.0001,
            'pretrained_weights':True,
        }

        # Initialize a new wandb run
        wandb.init(project='Grasp_planning', entity='shreyashgadgil007',config=config_defaults)
        wandb.run.name = 'Grasp_planning_exper:-'+'model: '+ str(wandb.config.Model)+' ;learning_rate: '+str(wandb.config.learning_rate)+ ' ;pretrained_weights:'+str(wandb.config.pretrained_weights)

        
        config = wandb.config
        Model = config.Model
        learning_rate = config.learning_rate
        pretrained_weights = config.pretrained_weights
        
        
        # Define the training parameters
        epochs = 10
        #learning_rate = 0.0001
    
        # Initialize the model, loss function, and optimizer
        #model = CNN_Position_Orientation(num_outputs=1).to(device)
        if Model == 'ResNet50':
            model = ResNet_Position_Orientation(num_outputs=1, freeze_weights=pretrained_weights).to(device)
        elif Model == 'EfficientNet':
            model = EfficientNet_Position_Orientation(num_outputs=1, freeze_weights=pretrained_weights).to(device)
    
        #model = Vision_Transformer_Position_Orientation(num_outputs=1).to(device)
        position_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
          total_loss = 0.0
          for batch_idx, (rgbd_images, positions, orientations) in enumerate(dataloader):
            # Forward pass
            rgbd_images = rgbd_images.to(device)
            positions = positions.to(device)
            #orientations = orientations.to(device)
            positions= positions.float()
            #orientations = orientations.float()
            pos_outputs = model(rgbd_images)
            #orient_outputs = model(rgbd_images)
            # Compute the loss
            loss_1 = position_criterion(pos_outputs, positions)
            #loss_2  = quaternion_loss(orient_outputs, orientations)
            #loss_2  = position_criterion(orient_outputs,orientations)
            print('loss1: ',loss_1)
            loss = loss_1
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch statistics
            total_loss += loss.item()
            '''if batch_idx % 10 == 9:  # Print every 10 batches
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, batch_idx+1, len(dataloader), total_loss / 10))
                total_loss = 0.0'''
          # Calculate accuracy after each epoch
          #Change the syntax such that accuracy function is faster     
          epoch_accuracy = calculate_accuracy(model, dataloader)
          epoch_accuracy_eval = calculate_accuracy(model, dataloader_eval)
          print('Epoch [{}/{}], Train_loss:{:.6f} Train_accuracy: {:.2f}%, Validation_accuracy: {:.2f}%'.format(epoch+1, epochs,  total_loss / len(dataloader), epoch_accuracy, epoch_accuracy_eval))
          print("-----------------------------------------------")
          wandb.log({"train_loss":total_loss / len(dataloader),"train_accuracy": epoch_accuracy ,"val_accuracy": epoch_accuracy_eval},)
          #emptying the cache after one complete run
          if epoch==epochs-1:
                    torch.cuda.empty_cache()

    sweep_id = wandb.sweep(sweep_config, entity='shreyashgadgil007', project="Grasp_planning")
    wandb.agent(sweep_id, function=sweep_train, count=120)
