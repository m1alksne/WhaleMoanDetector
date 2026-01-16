"""
Adapted from older version on August 15th 2025

@author: Michaela Alksne and Shane Andres

Contains helper functions for performing inference

"""

import yaml
import os
import torchvision.ops as ops
from PIL import Image
from spectrogram_functions import *

def predict_and_save_spectrograms(wav_file_path, model, model_name, device):
    '''
    For one wav file,
    1) Performs inference on each chunk
    2) Saves the spectrogram for any chunk containing detections
    3) Returns all detections in a list
    Inputs:
    - wav_file_path: location of wav file to run inference on
    - model: the model to use
    - model_name: the name of the model (logged in the model output)
    - device: the torch device to store data on
    Outputs:
    - output: a list of dicts, where each dict stores information about one detection in the model output format
    '''

    # loading spectrogram settings
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    thresholds = config['inference']['thresholds']
    nms_threshold = config['inference']['nms_threshold']
    spectrogram_folder = config['inference']['spectrogram_folder']
    categories = config['categories']
    categories_rev = {v: k for k, v in categories.items()}

    # generating spectrograms
    wav_file_name = os.path.splitext( os.path.basename(wav_file_path) )[0]
    wav_file_name = os.path.splitext( wav_file_path )[0] # removes .x for xwav files
    chunks, chunk_start_times, chunk_end_times, sr = chunk_audio(wav_file_path, device)  
    spectrograms = chunk_to_spectrogram(chunks, sr, device)
    
    t0 = chunk_start_times[0] 

    output = []
    for spectrogram, chunk_start_time in list(zip(spectrograms, chunk_start_times)):
        
        # run inference
        spectrogram_model = spectrogram.unsqueeze(0).unsqueeze(0).to(torch.float) / 255 # set shape to [N, C, H, W] and set range to to [0, 1]
        spectrogram_model = spectrogram_model.to(device)
        with torch.no_grad():
            prediction = model(spectrogram_model)[0]
        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']
       
        # apply non-maximum suppression (NMS)
        keep_indices = ops.nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # check if there are valid predictions (boxes) and save spectrograms containing predictions
        if len(boxes) == 0:
            continue
        
        saved = False
        # save detections to output
        for box, score, label in zip(boxes, scores, labels):
            category = categories_rev.get(label.item(), 'Unknown')
            if score.item() < thresholds.get(category, 0): # skips detections where score < threshold
                continue
            elif not saved: # saves spectrograms containing at least one detection with score > threshold
                spectrogram_file = os.path.basename(name_spectrogram_file(wav_file_name, chunk_start_time))
                spectrogram_path = os.path.join(spectrogram_folder, spectrogram_file)
                spectrogram_img = Image.fromarray(spectrogram.numpy())
                spectrogram_img.save(spectrogram_path)
                saved = True
                
            # get box time offsets
            x1, x2 = box[0].item(), box[2].item()
            y1, y2 = box[1].item(), box[3].item()
            start_offset_sec, end_offset_sec = pixel_to_time(x1), pixel_to_time(x2)
            # absolute seconds since WAV start (t0)
            chunk_start_sec = (chunk_start_time - t0).total_seconds()
            start_time_sec = chunk_start_sec + start_offset_sec
            end_time_sec   = chunk_start_sec + end_offset_sec
            start_time = chunk_start_time + timedelta(seconds=start_offset_sec)
            end_time = chunk_start_time + timedelta(seconds=end_offset_sec)
            
            output.append({
                'wav_file_path': wav_file_path,
                'model_no': model_name,
                'image_file_path': spectrogram_path,
                'label': category,
                'score': round(score.item(), 2),
                'start_time_sec': start_time_sec,
                'end_time_sec': end_time_sec,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'min_frequency': pixel_to_freq(y2),
                'max_frequency': pixel_to_freq(y1),
                'box_x1': x1,
                'box_x2': x2,
                'box_y1': y1,
                'box_y2': y2
                })
    
    return output