# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:25:24 2024

@author: Michaela ALksne

Make spectrograms and bounding box annotations for each spectrogram in a given wav file
- starts with "modififed annotations" which contain the start and end times of the annotations in seconds since the start of the wav file
- makes "chunks" for each 60 second window in the wav file (with 30 seconds of overlap)
- finds the annotation time stamps for each chunk and makes a spectrogram around them with bounding box annotations in the format:
    [xmin, ymin, xmax, ymax]
- saves the spectrograms and an annotation csv pointing to each spectrogram and its corresponding bounding box
- if multiple bounding boxes exist per spectrogram, the spectrogram annotation gets repeated row-wise
- loops through whatever files you point it to

- puts it on the GPU (5/9/24)
- makes/saves ALL spectrograms in a given wav file (7/26/24)
    
"""

import glob
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from make_spectrograms import make_spectrograms

directory_path = "L:/WhaleMoanDetector/labeled_data/modified_annotations/HARP/" # point to modified annotation files
all_files = glob.glob(os.path.join(directory_path, "*csv")) # path for all files
output_dir = "L:/WhaleMoanDetector/labeled_data/spectrograms/HARP"

for file in all_files:
    # Parse the unique part of the filename you want to use for naming
    unique_name_part = Path(file).stem  # Adjust index as needed
    annotations_df = pd.read_csv(file)
    # Call your function to process the annotations and generate spectrograms
    make_spectrograms(unique_name_part,annotations_df, output_dir, window_size=60, overlap_size=0)
    #make_spectrograms_pcen(unique_name_part,annotations_df, output_directory, window_size=60, overlap_size=0)




