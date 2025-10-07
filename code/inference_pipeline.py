"""
Adapted from older version on August 15th 2025

@author: Michaela Alksne and Shane Andres

Performs inference on a dataset, and saves:
1) Any spectrogram containing a detection inside of spectrogram_folder
2) All detections in the model output format to 'raw_detections.txt' inside of detections_folder

"""

import torch
import os
import yaml
from tqdm import tqdm
from call_context_filter import *
from model_functions import *
from inference_functions import *

# loading file paths and settings
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
wav_folder = config['inference']['wav_folder']
detections_folder = config['inference']['detections_folder']
model_folder = config['train']['model_folder']
CalCOFI_flag = config['spectrogram']['CalCOFI_flag']
categories = config['categories']

categories_rev = {v: k for k, v in categories.items()}
num_classes = len(categories) + 1


# !!! user input !!!
model_name = "your_model_name"
model_constructor = RCNN_ResNet_50      
eval_epoch = 29

model_path = f'{model_folder}/{model_name}/{model_name}_epoch_{eval_epoch}.pth'
model = model_constructor(num_classes)
checkpoint = torch.load(model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


# define data parameters
fieldnames = ['wav_file_path', 'model_no', 'image_file_path', 'label', 'score',
              'start_time_sec','end_time_sec','start_time','end_time',
              'min_frequency', 'max_frequency','box_x1', 'box_x2', 
              'box_y1', 'box_y2' ]
detections_path = os.path.join(detections_folder, 'raw_detections.txt')
with open(detections_path, mode='w', encoding='utf-8') as detections:
    detections.write('\t'.join(fieldnames) + '\n')
    

# loop over each wav file in folder
printout = "=============== RUNNING INFERENCE ==============="
print("=" * len(printout))
print(printout)
print("=" * len(printout) + "\n")

for root, _, files in os.walk(wav_folder):
    wav_files = [f for f in files if f.lower().endswith((".wav", ".x.wav"))] # filters out all non-wav files
    for wav_file in tqdm(wav_files):

        # run inference on spectrograms and save any that contain detections
        wav_file_path = os.path.join(root, wav_file)
        wav_file_name = os.path.splitext(wav_file)[0]
        wav_file_name = os.path.splitext(wav_file_name)[0] # remove .x for xwav files     
        tqdm.write(wav_file)
        output = predict_and_save_spectrograms(wav_file_path, model, model_name, device)     

        # write detections to TXT file
        for detection in output:
            # write each detection as a line in the txt file, tab-separated
            with open(detections_path, mode='a', encoding='utf-8') as detections:
                detections.write('\t'.join(str(detection[field]) for field in fieldnames) + '\n')

call_context_filter(detections_path)