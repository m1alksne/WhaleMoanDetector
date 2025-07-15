# -*- coding: utf-8 -*-
"""
Created on July 2nd 2025

@author: Shane Andres

# script to generate evaluation metrics after training a model.
# note that the new training script does this automatically as it trains!

"""


# !!! Imports !!!

import torch
from torch.utils.data import DataLoader
from AudioDetectionDataset import AudioDetectionData, AudioDetectionData_with_hard_negatives
from custom_collate import custom_collate
from validation import validation
import os
import yaml
from model_functions import *


# !!! User input !!!

model_name = "WMD_MobileNet_v3.1"
model_constructor = RCNN_MobileNet_v3
val_set_name = "CalCOFI_200808_test"
eval_epoch = 29
iou_threshold = 0.1


# !!! Loading file paths !!!

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_path = config['model_path']
model_folder = config['model_folder']
labeled_data_folder = config['labeled_data_folder']
evaluation_folder = config['evaluation_folder']
categories = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
num_classes = len(categories) + 1 # Five classes plus background


# !!! Model architecture !!!

validation_log_folder = f'{evaluation_folder}/{model_name}'
model_path = f'{model_folder}/{model_name}/{model_name}_epoch_{eval_epoch}.pth'
if(not os.path.isdir(validation_log_folder)):
    raise OSError(f'Model {model_name} eval folder does not exist')
model = model_constructor(num_classes)

checkpoint = torch.load(model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
val_batch_size = 4


# !!! Loading data !!!
          
val_set_file = val_set_name + ".csv"
val_loader = DataLoader(AudioDetectionData_with_hard_negatives(csv_file=f'{labeled_data_folder}/{val_set_file}'),
                      batch_size=val_batch_size,
                      shuffle = False,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if str(device) != "cuda":
    print("Using " + str(device))


# !!! Running eval !!!

model.to(device)
model.eval()
with torch.no_grad():
    PR_header = (
        f"\n========== Inference on {model_name} ==========\n"
        f'Pulling labels from: {val_set_file}\n'
        f'Using an IOU threshold of: {iou_threshold}\n')
    PR_output, AP_list = validation(val_loader, device, model, categories, iou_threshold=iou_threshold)
    PR_file_path = f"{validation_log_folder}/{val_set_name}_{model_name}_{int(100*iou_threshold)}_percent_iou.txt"
    with open(PR_file_path, "w") as f:
        f.write(PR_output)

    print(f"\n========== Inference on {model_name}: ==========")
    print(f'Pulling labels from: {val_set_file}')
    print(f'Using an IOU threshold of: {iou_threshold}')
    for category_name in categories:
        print(f"{category_name} AP: {AP_list[category_name]}")
    


    

 
    
    
    
        
        
        
        
        
        
        
        
