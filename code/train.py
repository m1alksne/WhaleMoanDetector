# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne

# script to train faster rCNN model 
# this is for the Sonobuoys!!!!

# the pretrained Faster R-CNN ResNet-50 model that we are going to use expects the input image tensor to be in the form [c, h, w], where:

# c is the number of channels, for RGB images its 3 (which is what I have rn)
# h is the height of the image
# w is the width of the image

"""


# !!! Imports !!!

import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from AudioDetectionDataset import AudioDetectionData, AudioDetectionData_with_hard_negatives
from custom_collate import custom_collate
from validation import validation
from torchvision.ops import box_iou
from model_functions import *
import os
import json
from tqdm import tqdm
import yaml
from datetime import datetime


# !!! Loading file paths !!!

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_path = config['model_path']
model_folder = config['model_folder']
labeled_data_folder = config['labeled_data_folder']
evaluation_folder = config['evaluation_folder']
categories = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
num_classes = len(categories) + 1 # Five classes plus background


# !!! Hyperparameters (User Input) !!!

model = RCNN_MobileNet_v3(num_classes)
train_set_name = "SOCAL34N_subsample"
val_set_name = "SOCAL34N_subsample"

lr = 0.001
momentum = 0.9
weight_decay = 0.0005
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
num_epochs = 10
train_batch_size = 4
val_batch_size = 1

train_set_file = train_set_name + ".csv"
val_set_file = val_set_name + ".csv"
model_log = {
    "model_name": "WMD_MobileNet_v3.2",
    "notes": "Testing out new model constructors.",
    "dataset": "SOCAL34N_subsample",
    "train_file": train_set_file,
    "val_file": val_set_file,
    "training_date": datetime.today().strftime("%Y-%m-%d"),
    "hyperparams": {
        "model": "Faster RCNN",
        "backbone": "MobileNet_v3",
        "epochs": num_epochs,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "learning_rate": lr,
        "momentum": momentum,
        "weight_decay": weight_decay
    }
}
model_name = model_log["model_name"]
model_log_folder = f'{model_folder}/{model_name}'
validation_log_folder = f'{evaluation_folder}/{model_name}'
if(os.path.isdir(model_log_folder) or os.path.isdir(validation_log_folder)):
    raise OSError(f'Model {model_name} already exists')
os.makedirs(model_log_folder)
os.makedirs(validation_log_folder)
with open(os.path.join(model_log_folder, "model_log.json"), "w") as f:
    json.dump(model_log, f, indent=4)


# !!! Loading data !!!

train_loader = DataLoader(AudioDetectionData_with_hard_negatives(csv_file=f'{labeled_data_folder}/{train_set_file}'),
                      batch_size=train_batch_size,
                      shuffle = True,
                      collate_fn = custom_collate, 
                      pin_memory = True if torch.cuda.is_available() else False)

                    
val_loader = DataLoader(AudioDetectionData_with_hard_negatives(csv_file=f'{labeled_data_folder}/{val_set_file}'),
                      batch_size=val_batch_size,
                      shuffle = False,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)
        
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if str(device) != "cuda":
    print("Using " + str(device))


# !!! Training loop !!!

model_printout = "=============== TRAINING " + model_name + " ==============="
print("=" * len(model_printout))
print(model_printout)
print("=" * len(model_printout) + "\n")
num_batches = int(len(train_loader.dataset) / train_batch_size)
model.to(device)
model.train()

progress_bar = tqdm(range(num_epochs))
for epoch in progress_bar:
    model.train()
    epoch_train_loss  = 0
    batch_idx = 0
    for data in train_loader:
        progress_bar.set_postfix(batch=f'{batch_idx}/{num_batches}')
        batch_idx = batch_idx + 1

        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device)) # sending each image from the dataloader to our cpu / gpu
            
            if d[1] is None:  # check if the target is None (hard negative example)
                # create a dummy target for hard negatives with label 0
                targ = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32).to(device),
                    'labels': torch.tensor([0], dtype=torch.int64).to(device)
                }
            else:
                targ = {
                    'boxes': d[1]['boxes'].to(device),
                    'labels': d[1]['labels'].to(device)
                }
                
        
            targets.append(targ)

        loss_dict = model(imgs,targets)
        loss = sum(v for v in loss_dict.values())
        epoch_train_loss += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    # save model from current epoch
    model_epoch_name = f"{model_name}_epoch_{epoch}"
    model_save_path = f'{model_log_folder}/{model_epoch_name}.pth'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_train_loss,
            }, model_save_path)


    # validation
    model.eval()
    progress_bar.set_postfix(batch='val...')
    with torch.no_grad():
        epoch_header = (
            f"\n========== Epoch {epoch} ==========\n"
            f"Training Loss: {epoch_train_loss:.4f}\n\n")
        PR_output, AP_list = validation(val_loader, device, model, categories)
        epoch_PR_output = epoch_header + PR_output
        PR_file_path = f"{validation_log_folder}/validation_{model_epoch_name}.txt"
        with open(PR_file_path, "w") as f:
            f.write(epoch_PR_output)

        tqdm.write(f"\n========== Epoch {epoch}: ==========\n")
        tqdm.write(f'Training loss: {epoch_train_loss}\n')
        for category_name in categories:
            tqdm.write(f"{category_name} AP: {AP_list[category_name]}")

        

 
    
    
    
        
        
        
        
        
        
        
        
