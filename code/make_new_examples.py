"""
Created on August 15th 2025

@author: Shane Andres

Contains function to create new examples from detections annotated in WMV

"""

import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from spectrogram_functions import *

def make_new_examples(detections_file_path, examples_file_path="new_examples.txt"):
    '''
    For a given set of audio files and a file containing detections,
    1. Divides each audio file up into chunks of length window_size
    2. Saves a spectrograms for each chunk to spectrogram_folder in config
    3. Finds detections belonging to each chunk
    4. Creates new examples (in the model input format) containing a row for each detection and its corresponding spectrogram 
       as well as rows for each negative example

    Inputs:
    - detections_file_path: the file in the model output format containing detections
    - examples_file_path: the file where new examples are saved to (in the model input format)
    '''

    printout = "=============== MAKING NEW EXAMPLES ==============="
    print("=" * len(printout))
    print(printout)
    print("=" * len(printout) + "\n")

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    wav_folder = config['inference']['wav_folder']
    spectrogram_folder = config['inference']['spectrogram_folder']

    # load detections
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(detections_file_path, sep='\t')
    format_string = "%Y-%m-%d %H:%M:%S"
    det_start_times = np.array([datetime.strptime(det_start_time, format_string) for det_start_time in df['start_time']])
    det_end_times = np.array([datetime.strptime(det_end_time, format_string) for det_end_time in df['end_time']])
    det_pr = np.array([pr for pr in df['pr']])

    # create dataframe for new examples
    new_df = pd.DataFrame(columns=['spectrogram_path', 'label', 'xmin', 'xmax', 'ymin', 'ymax'])

    # loop through each audio file
    for root, _, files in os.walk(wav_folder):
        wav_files = [f for f in files if f.lower().endswith((".wav", ".x.wav"))] # filters out all non-wav files
        for wav_file in tqdm(wav_files):
                
            # chunk audio and create spectrograms for each chunk
            wav_file_path = os.path.join(root, wav_file)
            wav_file_name = os.path.splitext(wav_file)[0]
            wav_file_name = os.path.splitext(wav_file_name)[0] # remove .x for xwav files
            tqdm.write(wav_file)
            
            chunks, chunk_start_times, chunk_end_times, sr = chunk_audio(wav_file_path, device)
            for chunk, chunk_start_time, chunk_end_time in zip(chunks, chunk_start_times, chunk_end_times):  

                spectrogram_file = name_spectrogram_file(wav_file_name, chunk_start_time)
                spectrogram_path = os.path.join(spectrogram_folder, spectrogram_file)
                if not os.path.exists(spectrogram_path):
                    spectrogram = chunk_to_spectrogram([chunk], sr, device)[0].numpy()
                    spectrogram = Image.fromarray(spectrogram)
                    spectrogram.save(spectrogram_path)
        
                # find all detections beginning and/or ending in the current chunk that are TP/TN
                det_mask = (chunk_start_time < det_start_times) * (det_start_times < chunk_end_time) + (chunk_start_time < det_end_times) * (det_end_times < chunk_end_time)
                det_mask = det_mask * ( (det_pr == 1) + (det_pr == 3) )
                det_indices = np.argwhere(det_mask).flatten()
        
                # if no detections in current chunk, add blank row to indicate hard negative
                if len(det_indices) == 0:
                    new_df.loc[len(new_df)] = {'spectrogram_path': spectrogram_path}
                    continue
        
                # for each detection in the current chunk, add a row to the new examples
                for row in det_indices:
                    new_df.loc[len(new_df)] = {'spectrogram_path': spectrogram_path,
                                        'label': df['label'][row],
                                        'xmin': time_to_pixel( (det_start_times[row] - chunk_start_time).total_seconds() ),
                                        'xmax': time_to_pixel( (det_end_times[row] - chunk_start_time).total_seconds() ),
                                        'ymin': freq_to_pixel( df['max_frequency'][row] ),  
                                        'ymax': freq_to_pixel( df['min_frequency'][row] )
                                        }


    new_df.to_csv(examples_file_path, sep='\t', index=False)