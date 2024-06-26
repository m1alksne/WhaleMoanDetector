# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne

script to run the required functions in the correct order to make inference predictions using trained model
"""

import librosa
import numpy as np
import torch
import os
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.colors import Normalize
import torch
import torchvision.ops as ops
from PIL import ImageOps
from datetime import datetime, timedelta
from IPython.display import display
import csv
import sys
sys.path.append(r"L:\Sonobuoy_faster-rCNN\code\PYTHON")
from AudioStreamDescriptor import WAVhdr
from inference_functions import extract_wav_start, chunk_audio, audio_to_spectrogram, predict_and_plot_on_spectrograms, apply_filters_to_predictions, visualize_filtered_predictions_on_spectrograms
from inference_functions import bounding_box_to_time_and_frequency, predictions_to_datetimes_frequencies_and_labels, save_filtered_predictions_on_spectrograms


# import all of the needed functions and process one deployment at a time.

wav_directory = 'E:/SOCAL26H/'
csv_file_path = 'L:/Sonobuoy_faster-rCNN/predictions/SOCAL26H/SOCAL26H.csv'
fieldnames = ['wav_file_path','image_file_path', 'label', 'score', 'start_time', 'end_time', 'min_frequency', 'max_frequency', 'xmin', 'ymin', 'xmax', 'ymax']
images_base_dir = os.path.dirname(csv_file_path)

# define variables 

window_size = 60
overlap_size = 5
time_per_pixel = 0.1  # Since hop_length = sr / 10, this simplifies to 1/10 second per pixel

# Load your trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # 5 classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)

model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load('L:/Sonobuoy_faster-rCNN/trained_model/Sonobuoy_model_epoch_14.pth'))
model.eval()


# Open the CSV file just once, and write headers
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Loop over each file in the directory or subdirectory
    for dirpath, dirnames, filenames in os.walk(wav_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
            # Full path to the WAV file
                file_path = os.path.join(dirpath, filename)
                
                # Extract the subdirectory name if exists
                subfolder = os.path.relpath(dirpath, wav_directory)
                
                if subfolder == '.':
                    # if subfolder exists, use this in audiobasepath name ( for saving images)
                    audio_basename = os.path.splitext(filename)[0]
                else:
                    # if no subfolder exists, use wav file name
                    audio_basename = os.path.splitext(os.path.basename(file_path))[0]
                # Extract the start datetime from the WAV file
                wav_start_datetime = extract_wav_start(file_path)  # Ensure this returns a datetime object

            # Process each WAV file as you have in your folder
                chunks = chunk_audio(file_path, window_size=window_size, overlap_size=overlap_size) # make wav chunks of given length and overlap
                spectrograms = audio_to_spectrogram(chunks, sr=librosa.get_samplerate(file_path)) # make spectrograms
                predictions = predict_and_plot_on_spectrograms(spectrograms, model, visualize=True)  # convert spectrograms to grayscale images and turn off visualization for batch processing
                filtered_predictions = apply_filters_to_predictions(predictions, nms_threshold=0.2, D_threshold=0.4, fortyHz_threshold=0.4, 
                                                  twentyHz_threshold=0.4, A_threshold=0.4, B_threshold=0.4)  # apply filters to predictions. All must have score above 0.5 

                chunk_start_times = [i * (window_size - overlap_size) for i in range(len(chunks))] # get start time of each spectrogram
                detailed_event_info = predictions_to_datetimes_frequencies_and_labels(
                    filtered_predictions, chunk_start_times, time_per_pixel, wav_start_datetime)
                image_paths = save_filtered_predictions_on_spectrograms(spectrograms, filtered_predictions, csv_file_path, audio_basename, chunk_start_times, window_size, overlap_size)

            # Write event details and image names to CSV
                for event, image_path in zip(detailed_event_info, image_paths):
                    event['wav_file_path'] = file_path
                    event['image_file_path'] = image_path
                    writer.writerow(event)

print(f"Detailed event information for all files saved to {csv_file_path}")






















