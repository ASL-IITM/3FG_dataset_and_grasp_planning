# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:32:13 2024

@author: Shreyash Gadgil
"""
#This script is to create a dataloader


import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


# Define custom dataset class
class RGBDDataset(Dataset):
    def __init__(self, rgb_dir, csv_file, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform

        # Get list of filenames in the directories
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.depth_files = sorted(os.listdir(depth_dir))

        # Ensure the number of RGB and depth images are the same
        assert len(self.rgb_files) == len(self.depth_files), "Number of RGB and depth images do not match"
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Load RGB and depth images
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path)
        #Converting depth images to (0,1) range
        depth_array = np.array(depth_img)
        scaled_depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()+1e-8)
        depth_img = Image.fromarray(scaled_depth_array)


        # Apply transformations if specified
        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)

        # Combine RGB and depth images into RGBD
        rgbd_img = torch.cat((rgb_img, depth_img), dim=0)
        # Get position and orientation values for all five sets
        positions = []
        orientations = []
        '''for i in range(5):
            idx_values = self.df.iloc[idx * 5 + i]
            positions.append(torch.tensor(idx_values[['X', 'Y', 'Z']].values))
            orientations.append(torch.tensor(idx_values[['x_q', 'y_q', 'z_q', 'w_q']].values))'''

        for i in range(1):
            idx_values = self.df.iloc[idx * 1 + i]
            positions.append(torch.tensor(idx_values[['X', 'Y', 'Z']].values))
            orientations.append(torch.tensor(idx_values[['x_q', 'y_q', 'z_q', 'w_q']].values))
        positions = torch.stack(positions)  # Convert list of tensors to a single tensor
        orientations = torch.stack(orientations)  # Convert list of tensors to a single tensor

        return rgbd_img, positions, orientations


