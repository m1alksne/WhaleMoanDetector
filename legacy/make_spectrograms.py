# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:20:20 2024

@author: Michaela Alksne

Generate noise-reduced, 60 second spectrograms
# default settings input, change 
"""
from pathlib import Path
import torchaudio
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from time_bbox import time_to_pixels
from freq_bbox import freq_to_pixels
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import librosa
import librosa.display

def make_spectrograms(unique_name_part,annotations_df, output_dir, window_size=60, overlap_size=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_list = []  # To collect annotations for each deployment

    # Group annotations by audio file
    grouped = annotations_df.groupby('audio_file')

    for audio_file_path, group in grouped:
        audio_path = Path(audio_file_path)  # Convert to Path object for easier handling
        audio_basename = audio_path.stem  # Get the basename without the extension
        print(audio_basename)
    
        # Load the audio file
        waveform, sr = torchaudio.load(audio_file_path)
        waveform = waveform.to('cuda') # move wavform to gpu for efficent processing
        samples_per_window = window_size * sr
        samples_overlap = overlap_size * sr

        # Process each chunk of the audio file
        for start_idx in range(0, waveform.shape[1], samples_per_window - samples_overlap):
            end_idx = start_idx + samples_per_window
            if end_idx > waveform.shape[1]:
                # If the remaining data is less than the window size, pad it with zeros
                padding_size = end_idx - waveform.shape[1]
                chunk = torch.nn.functional.pad(waveform[:, start_idx:], (0, padding_size))  # Pad the last part of the waveform
            else:
                chunk = waveform[:, start_idx:end_idx]
            # Compute STFT on GPU
            #S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr), return_complex=True)

            S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to('cuda'), return_complex=True)
            transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) #convert to dB and clip at 80dB
            S_dB_all = transform(torch.abs(S))
            
            S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz

            chunk_start_time = start_idx / sr
            chunk_end_time = chunk_start_time + window_size
            
            spectrogram_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_end_time)}.png"
            spectrogram_data = S_dB.cpu().numpy()  # Move data to CPU for image processing
            
            # normalize between fixed range (0-80 dB)
            #scaled_S_dB = (spectrogram_data - 0) / 80  # Scale between 0 and 1 based on the fixed range
            #S_dB_img = Image.fromarray((scaled_S_dB * 255).astype(np.uint8), 'L')  # Convert to 0-255 grayscale
            
            normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data)) 
            #normalize image b/t zero and one
            S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
            # Convert the processed data back to an image
            image = ImageOps.flip(S_dB_img)
            # Convert the image to a numpy array for processing
            img_array = np.array(image)
            
            # Threshold values for detecting AIS signal in frequency bands
            threshold_1 = 200  # Threshold for the first 10 pixel block
            threshold_2 = 180  # Threshold for the second 10 pixel block
            threshold_3 = 160  # Lower threshold for the third 10 pixel block

            # Gray value to replace the AIS signal
            gray_value = 128  # Mid-gray
            # Find the vertical white lines and gray them out
            # Loop through each column (time slice) in the spectrogram
          #  for col in range(img_array.shape[1]):
            # # Check first 10 pixel block (corresponding to the lowest frequency band)
           #     if np.sum(img_array[-10:, col]) > threshold_1 * 10:
                    # If the first 10 pixel block passes, check the second 10 pixel block
            #       if np.sum(img_array[-20:-10, col]) > threshold_2 * 10:
                        # If the second block passes, check the third block with a lower threshold
             #           if np.sum(img_array[-30:-20, col]) > threshold_3 * 10:
                            # If all conditions are met, gray out the entire column
              #              img_array[:, col] = gray_value  # Replace the entire column with gray 
            
            # Convert back to image
            final_image = Image.fromarray(img_array)
            # Flip the image vertically

            final_image.save(output_dir / spectrogram_filename)

            # Filter and adjust annotations for this chunk
            #relevant_annotations = group[(group['start_time'] >= chunk_start_time) & (group['end_time'] <= chunk_end_time)]
            #more inclusive
            relevant_annotations = group[(group['end_time'] > chunk_start_time) & (group['start_time'] < chunk_end_time)]
            # anything within the chunk.

            # Adjust annotation times relative to the start of the chunk
            for _, row in relevant_annotations.iterrows():
                
                # Calculate the overlap start and end times within the chunk
                adjusted_start_time = max(row['start_time'], chunk_start_time) - chunk_start_time
                adjusted_end_time = min(row['end_time'], chunk_end_time) - chunk_start_time
                
                # Calculate the duration of the annotation
                duration = adjusted_end_time - adjusted_start_time
                
                # Check if the annotation is at least 10 sec for A and B, 2 sec for D and 0.5 sec for 2 Hz and 40hz. 
                if (row['annotation'] in ['A', 'B'] and duration >= 10) or (row['annotation'] in ['D'] and duration >= 2) or (row['annotation'] in ['20Hz', '40Hz'] and duration >= 0.5):
                
                #adjusted_start_time = row['start_time'] - chunk_start_time
                #adjusted_end_time = row['end_time'] - chunk_start_time

                    xmin, xmax = time_to_pixels(adjusted_start_time, adjusted_end_time, S_dB.shape[1], window_size)
                    ymin, ymax = freq_to_pixels(row['low_f'], row['high_f'], S_dB.shape[0], sr, sr)

                    annotations_list.append({
                        "spectrogram_path": f"{output_dir}/{spectrogram_filename}",
                        "label": row['annotation'],
                        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
                
            # If there are no relevant annotations, add an entry with no label and bounding box
            if relevant_annotations.empty:
                annotations_list.append({
                    "spectrogram_path": f"{output_dir}/{spectrogram_filename}",
                    "label": None,
                    "xmin": None,
                    "ymin": None,
                    "xmax": None,
                    "ymax": None
                })

    # Convert annotations list to DataFrame and save as CSV
    df_annotations = pd.DataFrame(annotations_list)
    df_annotations.to_csv(f"{output_dir}/{unique_name_part}_annotations.csv", index=False)
    

