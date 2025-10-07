"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne and Shane Andres

Trains an object detection model to identify whale calls in spectrograms.

NOTE: pretrained Faster R-CNN ResNet-50 models expect the input image tensor to be in the form [c, h, w], where:
    c is the number of channels
    h is the height of the image
    w is the width of the image

"""

import torch
from torch.utils.data import DataLoader
from AudioDetectionDataset import AudioDetectionData_with_hard_negatives
from custom_collate import custom_collate
from validation import validation
from model_functions import *
import os
import json
from tqdm import tqdm
import yaml
from datetime import datetime


# loading file paths

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_folder = config['train']['model_folder']
labeled_data_folder = config['train']['labeled_data_folder']
evaluation_folder = config['train']['evaluation_folder']
categories = config['categories']
num_classes = len(categories) + 1 # Five classes plus background


# !!! user input !!!

model_name = "your_model_name"
model_constructor = RCNN_ResNet_50
train_set_file = "train_annotations.txt"
val_set_file = "val_annotations.txt"

lr = 0.001
momentum = 0.9
weight_decay = 0.0005
num_epochs = 30
train_batch_size = 4
val_batch_size = 1

model_log = {
    "model_name": model_name,
    "notes": "Testing out the code",
    "dataset": "train_annotations",
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

model = model_constructor(num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

model_log_folder = f'{model_folder}/{model_name}'
validation_log_folder = f'{evaluation_folder}/{model_name}'
if(os.path.isdir(model_log_folder)):
    raise OSError(f'Model {model_name} already exists')
os.makedirs(model_log_folder)
if(not os.path.isdir(validation_log_folder)):
    os.makedirs(validation_log_folder)
with open(os.path.join(model_log_folder, "model_log.json"), "w") as f:
    json.dump(model_log, f, indent=4)


# loading data

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


# training loop

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

        

 
    
    
    
        
        
        
        
        
        
        
        
