# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:32:13 2024

@author: Shreyash Gadgil
"""
#Torch dependencies
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchsummary import summary
import wandb
import numpy as np

#Classes dependencies
from Dataloader import RGBDDataset
from Vanilla_CNN_models import CNN_Position_Orientation, ResNet_Position_Orientation, ResNet_Position_Orientation_2, EfficientNet_Position_Orientation, EfficientNet_Position_Orientation_2
from Loss_and_accuracy import calculate_accuracy, quaternion_loss, angular_difference_loss, regression_metrics, compute_euler_angles

def save_outputs_to_csv(outputs, filename):
    flattened_outputs = []
    for batch_outputs in outputs:
        pos_1_outputs = batch_outputs['pos_1_outputs']
        pos_2_outputs = batch_outputs['pos_2_outputs']
        pos_3_outputs = batch_outputs['pos_3_outputs']
        predicted_d = batch_outputs['predicted_d']
        
        for pos_1, pos_2, pos_3, d in zip(pos_1_outputs, pos_2_outputs, pos_3_outputs, predicted_d):
            flattened_outputs.append([
                pos_1[0], pos_1[1], pos_1[2],
                pos_2[0], pos_2[1], pos_2[2],
                pos_3[0], pos_3[1], pos_3[2],
                d[0]
            ])
    df = pd.DataFrame(flattened_outputs, columns=[
        'pos_1_x', 'pos_1_y', 'pos_1_z',
        'pos_2_x', 'pos_2_y', 'pos_2_z',
        'pos_3_x', 'pos_3_y', 'pos_3_z',
        'd'
    ])

    df.to_csv(filename, index=False)

def evaluate_model(model, dataloader, device):
    model.eval()
    eval_outputs = []

    with torch.no_grad():
        for batch_idx, (rgbd_images, positions_1, positions_2, positions_3, centroid, orientation, d) in enumerate(dataloader):
            rgbd_images = rgbd_images.to(device)
            pos_1_outputs, pos_2_outputs, pos_3_outputs, predicted_d = model(rgbd_images)

            batch_outputs = {
                'pos_1_outputs': pos_1_outputs.detach().cpu().numpy(),
                'pos_2_outputs': pos_2_outputs.detach().cpu().numpy(),
                'pos_3_outputs': pos_3_outputs.detach().cpu().numpy(),
                'predicted_d': predicted_d.detach().cpu().numpy()
            }
            eval_outputs.append(batch_outputs)

    return eval_outputs

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
          'name': 'train_loss',
          'goal': 'minimize'  
        },
        'parameters': {
            'model': {
                'values': ['ResNet50']
            },
            'learning_rate': {
                'values': [0.00001]
            },
            
            'loss': {
                'values': ['MAE']
            },
            'pretrained_weights':{
                'values':[False]
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
    rgb_dir = 'D:\\Grasp planning\\egad_train_set_0.12\\masked_rgb'
    depth_dir = 'D:\\Grasp planning\\egad_train_set_0.12\\masked_depth'
    csv_file = 'D:\\Grasp planning\\merged_top_1_entries_train_0.12.csv'
    dataset = RGBDDataset(rgb_dir,csv_file, depth_dir, transform=transform)

    rgb_dir_eval = 'D:\\Grasp planning\\egad_eval_set_0.12\\masked_rgb'
    depth_dir_eval = 'D:\\Grasp planning\\egad_eval_set_0.12\\masked_depth'
    csv_file_eval = 'D:\\Grasp planning\\merged_top_1_entries_eval_0.12.csv'
    dataset_eval = RGBDDataset(rgb_dir_eval,csv_file_eval, depth_dir_eval, transform=transform)
    
    # Initialize DataLoader
    batch_size = 128
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
            'loss':'MAE',
            'pretrained_weights':False,
        }

        # Initialize a new wandb run
        wandb.init(project='Grasp_planning_3', entity='shreyashgadgil007',config=config_defaults)
        wandb.run.name = 'Grasp_planning_exper:-'+'model: '+ str(wandb.config.Model)+' ;loss: '+str(wandb.config.loss)+ ' ;learning_rate: '+str(wandb.config.learning_rate)+ ' ;pretrained_weights:'+str(wandb.config.pretrained_weights)
        
        config = wandb.config
        Model = config.Model
        learning_rate = config.learning_rate
        pretrained_weights = config.pretrained_weights
        loss = config.loss
        
        
        # Define the training parameters
        epochs = 50
    
        # Initialize the model, loss function, and optimizer
        #model = CNN_Position_Orientation(num_outputs=1).to(device)
        if Model == 'ResNet50':
            model = ResNet_Position_Orientation_2(num_outputs=1, freeze_weights=pretrained_weights).to(device)
        elif Model == 'EfficientNet':
            model = EfficientNet_Position_Orientation_2(num_outputs=1, freeze_weights=pretrained_weights).to(device)
    
        if loss == 'MSE':
            position_criterion = nn.MSELoss()
        elif loss == 'MAE':
            position_criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
          total_loss = 0.0
          for batch_idx, (rgbd_images, positions_1, positions_2, positions_3, centroid, orientation, d) in enumerate(dataloader):
            # Forward pass
            rgbd_images = rgbd_images.to(device)
            positions_1 = positions_1.to(device)
            positions_2 = positions_2.to(device)
            positions_3 = positions_3.to(device)
            centroid = centroid.to(device)
            orientation = orientation.to(device)
            d = d.to(device)
            
            positions_1 = positions_1.float()
            positions_2 = positions_2.float()
            positions_3 = positions_3.float()
            centroid = centroid.float()
            orientation = orientation.float()
            d = d.float()
            
            #expected_centroid = (positions_1 + positions_2 + positions_3)/3
            
            pos_1_outputs, pos_2_outputs, pos_3_outputs, centroid_output, orientation_output, d_output = model(rgbd_images)
           
            #predicted_centroid = (pos_1_outputs + pos_2_outputs + pos_3_outputs)/3
            #orient_outputs = model(rgbd_images)
            # Compute the loss
            loss_1 = position_criterion(pos_1_outputs, positions_1)
            loss_2 = position_criterion(pos_2_outputs, positions_2)
            loss_3 = position_criterion(pos_3_outputs, positions_3)
            loss_4 = position_criterion(centroid_output, centroid)
            loss_5 = position_criterion(orientation_output, orientation)
            loss_6 = position_criterion(d_output, d)
            #loss_2  = quaternion_loss(orient_outputs, orientations)
            #loss_2  = angular_difference_loss(orient_outputs, orientations)
            #loss_2  = position_criterion(orient_outputs,orientations)
            print('loss1: ',loss_1)
            print('loss2: ',loss_2)
            print('loss3: ',loss_3)
            
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + 0.001*loss_6 
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch statistics
            total_loss += loss.item()
          # Calculate accuracy after each epoch
          #Change the syntax such that accuracy function is faster     
          epoch_pos_1_rmse, epoch_pos_1_r2, epoch_pos_2_rmse, epoch_pos_2_r2, epoch_pos_3_rmse, epoch_pos_3_r2 = calculate_accuracy(model, dataloader)
          
          epoch_val_pos_1_rmse, epoch_val_pos_1_r2, epoch_val_pos_2_rmse, epoch_val_pos_2_r2, epoch_val_pos_3_rmse, epoch_val_pos_3_r2  = calculate_accuracy(model, dataloader_eval)
          
          print('Epoch [{}/{}], Train_loss:{:.4f} Train pos_1_rmse,r2: {:.4f},{:.4f}; Train pos_2_rmse,r2: {:.4f},{:.4f}; Train pos_3_rmse,r2: {:.4f},{:.4f};'.format(epoch+1, epochs,  total_loss / len(dataloader), epoch_pos_1_rmse, epoch_pos_1_r2, epoch_pos_2_rmse, epoch_pos_2_r2, epoch_pos_3_rmse, epoch_pos_3_r2))
          
          print('Val_pos_1_rmse,r2: {:.4f},{:.4f}; Val pos_2_rmse,r2: {:.4f},{:.4f}; Val pos_3_rmse,r2: {:.4f},{:.4f};'.format(epoch_val_pos_1_rmse, epoch_val_pos_1_r2, epoch_val_pos_2_rmse, epoch_val_pos_2_r2, epoch_val_pos_3_rmse, epoch_val_pos_3_r2))
          print("-----------------------------------------------")
          wandb.log({"train_loss":total_loss / len(dataloader),"train_pos_1_rmse": epoch_pos_1_rmse ,"train_pos_1_r2": epoch_pos_1_r2 ,
                     "train_pos_2_rmse": epoch_pos_2_rmse ,"train_pos_2_r2": epoch_pos_2_r2 ,
                     "train_pos_3_rmse": epoch_pos_3_rmse ,"train_pos_3_r2": epoch_pos_3_r2, 
                     "val_pos_1_rmse": epoch_val_pos_1_rmse ,"val_pos_1_r2": epoch_val_pos_1_r2 ,
                     "val_pos_2_rmse": epoch_val_pos_2_rmse ,"val_pos_2_r2": epoch_val_pos_2_r2 ,
                     "val_pos_3_rmse": epoch_val_pos_3_rmse ,"val_pos_3_r2": epoch_val_pos_3_r2})
          #emptying the cache after one complete run
          if epoch==epochs-1:
                    torch.cuda.empty_cache()
        final_model_path = 'D:\\Grasp planning network\\masked_model.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f'Final model saved to {final_model_path}')
        # Save the evaluation dataset outputs
# =============================================================================
#         eval_outputs = evaluate_model(model, dataloader_eval, device)
#         
#         if loss == 'MSE':
#             save_outputs_to_csv(eval_outputs, 'eval_MSE_outputs.csv')
#         elif loss == 'MAE':
#             save_outputs_to_csv(eval_outputs, 'eval_MAE_outputs.csv')
# =============================================================================



    sweep_id = wandb.sweep(sweep_config, entity='shreyashgadgil007', project="Grasp_planning_3")
    wandb.agent(sweep_id, function=sweep_train, count=1)
