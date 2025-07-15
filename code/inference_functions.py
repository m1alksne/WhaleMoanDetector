# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne

All of the functions needed to run inference data processing pipeline...
"""

import librosa
import numpy as np
import torch
import torchaudio
import torchvision
import os
import csv
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.colors import Normalize
import torchvision.ops as ops
from PIL import ImageOps
from datetime import datetime, timedelta
from IPython.display import display
from AudioStreamDescriptor import WAVhdr, XWAVhdr
#from xwav_functions import get_datetime
from datetime import timedelta


# helper function uses WAVhdr to read wav file header info and extract wav file start time as a datetime object
def extract_wav_start(path):
    
    if path.endswith('.x.wav'):
        xwav_hdr= XWAVhdr(path)
        return xwav_hdr.dtimeStart
    if path.endswith('.wav'):
        wav_hdr = WAVhdr(path)
        return wav_hdr.start
    
def get_datetime(xwav_time: float, xwav):
    if not isinstance(xwav, XWAVhdr):
        xwav = XWAVhdr(xwav)
    num_channels = xwav.xhd['NumChannels']
    bits_per_sample = xwav.xhd['BitsPerSample']
    sample_rate = xwav.xhd['SampleRate']
    bytes_per_sample = bits_per_sample / 8
    samples_per_raw = [int(b / (num_channels * bytes_per_sample)) for b in xwav.xhd['byte_length']]
    durations = [s / sample_rate for s in samples_per_raw]
    elapsed = 0
    for i, dur in enumerate(durations):
        if xwav_time < elapsed + dur:
            offset = xwav_time - elapsed
            return xwav.raw['dnumStart'][i] + timedelta(seconds=offset)
        elapsed += dur
    raise ValueError(f"xwav_time {xwav_time:.2f}s exceeds duration of file ({elapsed:.2f}s).")
    
    
# function to chunk xwav audio into non overlapping 60 second chunks from raw files
def chunk_audio_from_xwav_raw_headers(xwav_path, waveform, device):
    """
   Splits an x.wav audio tensor into 60-second chunks, using raw header timing to track real gaps.
   Pads the last chunk if necessary and respects missing raw files (non-contiguous blocks).
   
   Parameters:
   - xwav_path: str, path to .x.wav file
   - waveform: torch.Tensor, loaded audio [channels, samples]
   - device: torch.device
   
   Returns:
   - chunks: list of 60-second audio chunks [channels, chunk_samples]
   - sr: int, sample rate
   - chunk_times: list of datetime objects marking chunk start times
   """
    xwav = XWAVhdr(xwav_path)
    sr = xwav.xhd['SampleRate']
    n_channels = xwav.xhd['NumChannels']
    bps = xwav.xhd['BitsPerSample']
    bytes_per_sample = bps / 8
    # Compute number of samples per raw file
    samples_per_raw = [int(b / (n_channels * bytes_per_sample)) for b in xwav.xhd['byte_length']]
    dnum_start = xwav.raw['dnumStart'] # datetime list of raw file starts
    chunk_sec = 60
    chunk_samples = chunk_sec * sr

    chunks = []
    chunk_times = []
    # Initialize empty 2D buffer: [channels, 0 samples]
    buffer = torch.empty((waveform.shape[0], 0), dtype=waveform.dtype).to(device)
    current_time = None
    sample_ptr = 0
    prev_end_time = None

    for i, n_samples in enumerate(samples_per_raw):
        raw_time = dnum_start[i]
        raw_audio = waveform[:, sample_ptr:sample_ptr + n_samples]
        sample_ptr += n_samples
        # If there's a time gap > 1s, flush and pad the current buffer
        if prev_end_time and (raw_time - prev_end_time).total_seconds() > 1:
            if buffer.shape[1] > 0:
                padded = torch.nn.functional.pad(buffer, (0, chunk_samples - buffer.shape[1]), 'constant', 0)
                chunks.append(padded)
                chunk_times.append(current_time)
            buffer = raw_audio
            current_time = raw_time
        # No gap → concatenate new audio onto buffer
        else:
            if buffer.shape[1] == 0:
                current_time = raw_time
            buffer = torch.cat((buffer, raw_audio), dim=1)
        # While buffer contains 60s or more → extract chunks
        while buffer.shape[1] >= chunk_samples:
            chunks.append(buffer[:, :chunk_samples])
            chunk_times.append(current_time)
            buffer = buffer[:, chunk_samples:]
            current_time += timedelta(seconds=chunk_sec)

        prev_end_time = raw_time + timedelta(seconds=n_samples / sr)
    # Final chunk (pad if needed)
    if buffer.shape[1] > 0:
        padded = torch.nn.functional.pad(buffer, (0, chunk_samples - buffer.shape[1]), 'constant', 0)
        chunks.append(padded)
        chunk_times.append(current_time)

    return chunks, sr, chunk_times
   
# Function to load wav file audio file and chunk it into overlapping windows

def chunk_audio(audio_file_path, device, window_size=60, overlap_size=5):
    # Load audio file
    waveform, sr = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)  # Move waveform to GPU for efficient processing
    samples_per_window = window_size * sr
    samples_overlap = overlap_size * sr

    # Calculate the number of chunks
    chunks = []
    
    for start in range(0, waveform.shape[1], samples_per_window - samples_overlap):
        end = start + samples_per_window
        # If the last chunk is smaller than the window size, pad it with zeros
        if end > waveform.shape[1]:
            y_pad = torch.nn.functional.pad(waveform[:, start:], (0, end - waveform.shape[1]), mode='constant')
            chunks.append(y_pad)
        else:
            chunks.append(waveform[:, start:end])
   # chunk_start_times = [i * (window_size - overlap_size) for i in range(len(chunks))]
    chunk_start_times = [start / sr for start in range(0, waveform.shape[1], samples_per_window - samples_overlap)]

    return chunks, sr, chunk_start_times

# Function to convert audio chunks to spectrograms
def audio_to_spectrogram(chunks, sr, device): # these are default fft and hop_length, this is dyamically adjusted depending on the sr. 
    spectrograms = []
   
    for chunk in chunks:
        # Use librosa to compute the spectrogram
        S = torch.stft(chunk[0], n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to(device), return_complex=True)
        transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) #convert to dB and clip at 80dB
        S_dB_all = transform(torch.abs(S))
        S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz
        spectrograms.append(S_dB.cpu().numpy())
    return spectrograms

      
                
def predict_and_save_spectrograms(spectrograms, model, CalCOFI_flag, device, csv_file_path, wav_file_path, wav_start_time, audio_basename, 
                                  chunk_start_times, window_size, overlap_size, inverse_label_mapping, time_per_pixel, is_xwav,
                                  A_thresh, B_thresh, D_thresh, TwentyHz_thresh, FourtyHz_thresh,
                                  freq_resolution=1, start_freq=10, max_freq=150):
    predictions = []
    csv_base_dir = os.path.dirname(csv_file_path)
    
    # Threshold dictionary for easy access by label name
    thresholds = {
        'A': A_thresh,
        'B': B_thresh,
        'D': D_thresh,
        '20Hz': TwentyHz_thresh,
        '40Hz': FourtyHz_thresh
    }
    
    # Zip spectrograms and start times
    data = list(zip(spectrograms, chunk_start_times))
    for spectrogram_data, chunk_start_time in data:
        # Normalize spectrogram and convert to tensor
        normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data))  
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L')
        image = ImageOps.flip(S_dB_img)
        # Convert the image to a numpy array for processing
        img_array = np.array(image)
        
        if CalCOFI_flag:
        # CalCOFI Sonobuoy cleanup for AIS

            threshold_1 = 200  # Threshold for the first 10 pixel block
            threshold_2 = 180  # Threshold for the second 10 pixel block
            threshold_3 = 160  # Lower threshold for the third 10 pixel block
            # Gray value to replace the AIS signal
            gray_value = 128  # Mid-gray
            # Find the vertical white lines and gray them out
            # Loop through each column (time slice) in the spectrogram
            for col in range(img_array.shape[1]):  # Loop through each column
            # Check first 10 pixel block (corresponding to the lowest frequency band)
                if np.sum(img_array[-10:, col]) > threshold_1 * 10:
                    # If the first 10 pixel block passes, check the second 10 pixel block
                    if np.sum(img_array[-20:-10, col]) > threshold_2 * 10:
                        # If the second block passes, check the third block with a lower threshold
                        if np.sum(img_array[-30:-20, col]) > threshold_3 * 10:
                            # If all conditions are met, gray out the entire column
                            img_array[:, col] = gray_value  # Replace the entire column with gray 
        
        
        # Convert back to image
        final_image = Image.fromarray(img_array)
    
        # Convert to tensor
        S_dB_tensor = F.to_tensor(final_image).unsqueeze(0).to(device)        
        # Run prediction
        model.eval()
        with torch.no_grad():
            prediction = model(S_dB_tensor)
        
        # Extract prediction results
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']
       
        # Apply Non-Maximum Suppression (NMS)
        keep_indices = ops.nms(boxes, scores, 0.05)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        # Check if there are valid predictions (boxes)
        if len(boxes) > 0:
            if is_xwav:
                chunk_start_datetime = chunk_start_time  # already datetime
                chunk_start_offset_sec = None  # we don't track this in xwavs now
            else:
                chunk_start_offset_sec = chunk_start_time  # float
                chunk_start_datetime = wav_start_time + timedelta(seconds=chunk_start_time)

            chunk_end_datetime = chunk_start_datetime + timedelta(seconds=window_size)
            
            # Save the spectrogram image
            if is_xwav:
                timestamp_str = chunk_start_datetime.strftime('%Y%m%dT%H%M%S')
                image_filename = f"{audio_basename}_{timestamp_str}.png"
            else:
                image_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_start_time + window_size)}.png"
    
            image_path = os.path.join(csv_base_dir, image_filename)
            final_image.save(image_path)

            # Iterate through detections
        for box, score, label in zip(boxes, scores, labels):
            textual_label = inverse_label_mapping.get(label.item(), 'Unknown')
            if score.item() < thresholds.get(textual_label, 0):
                continue

            # Get box time offsets
            start_offset_sec = box[0].item() * time_per_pixel
            end_offset_sec = box[2].item() * time_per_pixel

            if is_xwav:
                start_datetime = chunk_start_datetime + timedelta(seconds=start_offset_sec)
                end_datetime = chunk_start_datetime + timedelta(seconds=end_offset_sec)
                start_time_sec = None
                end_time_sec = None
            else:
                start_time_sec = start_offset_sec + chunk_start_offset_sec
                end_time_sec = end_offset_sec + chunk_start_offset_sec
                start_datetime = wav_start_time + timedelta(seconds=start_time_sec)
                end_datetime = wav_start_time + timedelta(seconds=end_time_sec)

            predictions.append({
                'image_file_path': image_path,
                'label': textual_label,
                'score': round(score.item(), 2),
                'start_time_sec': start_time_sec,
                'end_time_sec': end_time_sec,
                'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'min_frequency': round(max_freq - box[3].item() * freq_resolution),
                'max_frequency': round(max_freq - box[1].item() * freq_resolution),
                'box_x1': box[0].item(),
                'box_x2': box[2].item(),
                'box_y1': box[1].item(),
                'box_y2': box[3].item()
                })
    
    return predictions












