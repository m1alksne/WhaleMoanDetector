# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne

script to run the required functions in the correct order to make predictions using trained model

NOTE: Untested changes added on 6/25, moving the import model function to the inference_functions script.
      This should not impact the functionality of this pipeline.
"""

import librosa
import numpy as np
import torch
import os
import torchvision
import torchaudio
from AudioStreamDescriptor import WAVhdr, XWAVhdr
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.colors import Normalize
import torch
import pandas as pd
import torchvision.ops as ops
from PIL import ImageOps
from datetime import datetime, timedelta
from IPython.display import display
import csv
import yaml
from inference_functions import *
from call_context_filter import *

# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# Access the configuration variables
wav_directory = config['wav_directory']
txt_file_path = config['txt_file_path']
model_path = config['model_path']
CalCOFI_flag = config['CalCOFI_flag']

A_thresh=0
B_thresh=0
D_thresh=0
TwentyHz_thresh=0
FourtyHz_thresh=0

# Define spectrogram and data parameters
fieldnames = ['wav_file_path', 'model_no', 'image_file_path', 'label', 'score',
              'start_time_sec','end_time_sec','start_time','end_time',
              'min_frequency', 'max_frequency','box_x1', 'box_x2', 
              'box_y1', 'box_y2' ]
model_name = os.path.basename(model_path)
visualize_tf = False
label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
window_size = 60
overlap_size = 0
time_per_pixel = 0.1  # Since hop_length = sr / 10, this simplifies to 1/10 second per pixel

# Loading model
num_classes = len(label_mapping) + 1 # All desired call types + one background class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = import_RCNN(model_path, num_classes, device)
model.eval()

# Open the TXT file and write headers
with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
    # Write headers with tab as a delimiter
    txtfile.write('\t'.join(fieldnames) + '\n')

    # Loop over each file in the directory or subdirectory
    for dirpath, dirnames, filenames in os.walk(wav_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                # Full path to the WAV file
                wav_file_path = os.path.join(dirpath, filename)

                # Extract the subdirectory name if it exists
                subfolder = os.path.relpath(dirpath, wav_directory)

                if subfolder == '.':
                    audio_basename = os.path.splitext(filename)[0]
                    print(audio_basename)
                else:
                    audio_basename = os.path.splitext(os.path.basename(wav_file_path))[0]
                    print(audio_basename)

                # Extract the start datetime from the WAV file
                wav_start_time = extract_wav_start(wav_file_path)  # Ensure this returns a datetime object
                
                if filename.endswith('.x.wav'):
                    xwav = XWAVhdr(wav_file_path)
                    waveform, sr = torchaudio.load(wav_file_path)
                    waveform = waveform.to(device)
                    chunks, sr, chunk_start_times = chunk_audio_from_xwav_raw_headers(wav_file_path, waveform, device)
                    is_xwav = True
                else:
                    chunks, sr, chunk_start_times = chunk_audio(wav_file_path, device, window_size=60, overlap_size=0)
                    is_xwav = False
                
               
                # Process each WAV file as you have in your folder
                spectrograms = audio_to_spectrogram(chunks, sr, device)
                # Predict on spectrograms and save images and data for positive detections
                predictions = predict_and_save_spectrograms(
                    spectrograms, model, CalCOFI_flag, device, txt_file_path, wav_file_path, wav_start_time,
                    audio_basename, chunk_start_times, window_size, overlap_size,
                    inverse_label_mapping, time_per_pixel, is_xwav, A_thresh, B_thresh, D_thresh, 
                    TwentyHz_thresh, FourtyHz_thresh,
                    freq_resolution=1, start_freq=10, max_freq=150)
                

                # Write event details to the TXT file
                for event in predictions:
                    event['wav_file_path'] = wav_file_path
                    event['model_no'] = model_name
                    # Write each event as a line in the txt file, tab-separated
                    txtfile.write('\t'.join(str(event[field]) for field in fieldnames) + '\n')

print('Predictions complete')

print('Running call context filter')

call_context_filter(txt_file_path)















