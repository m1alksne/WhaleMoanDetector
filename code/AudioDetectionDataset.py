# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne

make a custom Dataset class for my spectrograms
for the sonobuoys
"""

import os
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim


class AudioDetectionData(Dataset):
    
    def __init__(self, csv_file):
        
        ext = os.path.splitext(csv_file)[1]
        if ext == '.csv':
            self.data = pd.read_csv(csv_file)
        elif ext == '.txt':
            self.data = pd.read_csv(csv_file, sep='\t')
        else:
            raise OSError("Error: unknown dataset file type")
        self.label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A NE Pacific': 4, 'B NE Pacific': 5}
        
        # Group data by 'ImageName' and aggregate all boxes and labels for each image
        self.grouped_data = self.data.groupby('spectrogram_path')
        self.unique_image_names = self.data['spectrogram_path'].unique()

        
    def __len__(self):
        # The length is the number of unique images
        return len(self.unique_image_names)
    
    def __getitem__(self, idx):
        # Use the unique image name to access the group
        image_name = self.unique_image_names[idx]
        image_data = self.grouped_data.get_group(image_name)
        
        # Get image path
        img_path = image_data.iloc[0]['spectrogram_path']
        img = Image.open(img_path).convert('L')
        
        # Get boxes and labels
        boxes = image_data[['xmin', 'ymin', 'xmax', 'ymax']].values.astype('float')
        labels = image_data['label'].tolist()
        label_encoded = [self.label_mapping[label] for label in labels]
        label_encoded = torch.tensor(label_encoded, dtype=torch.int64)
        
        # Create target dict
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = label_encoded
        
        return T.ToTensor()(img), target
    
    
class AudioDetectionData_with_hard_negatives(Dataset):
    
    def __init__(self, csv_file):
        
        ext = os.path.splitext(csv_file)[1]
        if ext == '.csv':
            self.data = pd.read_csv(csv_file)
        elif ext == '.txt':
            self.data = pd.read_csv(csv_file, sep='\t')
        else:
            raise OSError("Error: unknown dataset file type")
        self.label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
        
        # Group data by 'spectrogram_path' and aggregate all boxes and labels for each image
        self.grouped_data = self.data.groupby('spectrogram_path')
        self.unique_image_names = self.data['spectrogram_path'].unique()

        
    def __len__(self):
        # The length is the number of unique images
        return len(self.unique_image_names)
    
    def __getitem__(self, idx):
        # Use the unique image name to access the group
        image_name = self.unique_image_names[idx]
        image_data = self.grouped_data.get_group(image_name)
        
        # Get image path
        img_path = image_data.iloc[0]['spectrogram_path']
        #print(img_path)
        img = Image.open(img_path).convert('L')
        
        # Check for NaN labels (indicating hard negative example)
        if pd.isnull(image_data.iloc[0]['label']):
            return T.ToTensor()(img), None, img_path  
        
        # Get boxes and labels
        boxes = image_data[['xmin', 'ymin', 'xmax', 'ymax']].values.astype('float')
        labels = image_data['label'].tolist()
        label_encoded = [self.label_mapping[label] for label in labels]
        label_encoded = torch.tensor(label_encoded, dtype=torch.int64)
        
        
        # Create target dict
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = label_encoded
        #print(img_path)
        
        return T.ToTensor()(img), target, img_path
