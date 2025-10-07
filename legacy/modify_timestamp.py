# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:35:46 2024

@author: Michaela Alksne 

Script to run when modifying Triton logger annotation excel datasheets
converts xls to csv containing the audio file path, the annotation label, the frequency bounds, and time bounds. 
saves new csv in "modified annotations subfolder"
wav is the audio file
start time = start time of call in number of seconds since start of wav 
end time = end time of call in number of seconds since start of wav

"""

from datetime import datetime
import os
import glob
import sys
from AudioStreamDescriptor import WAVhdr
from modify_timestamp_function import modify_annotations
import random
import pandas as pd
import numpy as np

directory_path = "L:/WhaleMoanDetector/labeled_data/modified_annotations/backup" # point to original logger files
all_files = glob.glob(os.path.join(directory_path,'SOCAL_H_65_log.csv')) # path for all files

new_base_path = 'L:/SOCAL_H_65_180709_051300_df100.x' # path to change to 

# make a subfolder for saving modified logs 
subfolder_name = "modified_annotations"
# Create the subfolder if it doesn't exist
subfolder_path = os.path.join(directory_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# loop through all annotation files and save them in subfolder "modified_annotations"

for file in all_files:
    data = pd.read_csv(file)
    
    # if any(data['Input file'].str.contains('DCPP01A_d01_121106_083945.d100.x.wav')):
        
    #     # 'DCPP01A_d01_121106_083945.d100.x.wav' is missing a chunk of data between these bounds:
    #     date1 = datetime(2012, 11, 6, 8, 41, 11)
    #     date2 = datetime(2012, 11, 7, 2, 0, 0)
    #     # Calculate the difference in seconds
    #     seconds_difference = (date2 - date1).total_seconds()

    #     mask = data['Input file'].str.contains('DCPP01A_d01_121106_083945.d100.x.wav')
    #     subset_df = modify_annotations(data, new_base_path)
    #     subset_df.loc[mask, 'start_time'] -= seconds_difference
    #     subset_df.loc[mask, 'end_time'] -= seconds_difference
    #     subset_df.reset_index(drop=True)
        
    # elif any(data['Input file'].str.contains('SOCAL26H_d01_080604_173900.d100.x.wav')):
        
    #     # 'SOCAL26H_d01_080604_173900.d100.x.wav' is missing a chunk of data between these bounds:
    #     date1 = datetime(2008, 6, 4, 17, 40, 15)
    #     date2 = datetime(2008, 6, 5, 0, 0, 0)
    #     # Calculate the difference in seconds
    #     seconds_difference = (date2 - date1).total_seconds()

    #     mask = data['Input file'].str.contains('SOCAL26H_d01_080604_173900.d100.x.wav')
    #     subset_df = modify_annotations(data, new_base_path)
    #     subset_df.loc[mask, 'start_time'] -= seconds_difference
    #     subset_df.loc[mask, 'end_time'] -= seconds_difference
    #     subset_df.reset_index(drop=True)
    # else:
        
        
    subset_df = modify_annotations(data, new_base_path)
    
   
    filename = os.path.basename(file)
    new_filename = filename.replace('.xls', '_modification.csv')
     # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    subset_df.to_csv(save_path, index=False)


