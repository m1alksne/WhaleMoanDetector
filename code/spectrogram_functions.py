"""
Adapted from older version on August 15th 2025

@author: Michaela Alksne and Shane Andres

Contains helper functions for working with spectrograms

"""

import torch
import torchaudio
import yaml
from datetime import timedelta
from AudioStreamDescriptor import WAVhdr, XWAVhdr


def chunk_audio(audio_file_path, device):
    '''
    Breaks C-channel audio files into chunks of samples.
    Inputs:
    - audio_file_path: absolute path to audio file
    - device: the torch device to store data on
    Outputs:
    - chunks: A list containing chunks of audio samples, with each chunk of shape [C, N] for C channels and N samples
      Note that any chunks that are too short to fill the desired window size are zero padded to length.
    - chunk_start_times: the start time of each chunk in datetime format
    - chunk_end_times: the end time of each chunk in datetime format
    - sr: audio file sample rate
    '''

    if audio_file_path.endswith('.x.wav'):
        return chunk_audio_xwav(audio_file_path, device)
    elif not audio_file_path.endswith('.wav'):
        raise OSError('Error: unsupported audio file type')

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    window_size = config['spectrogram']['window_size']
    overlap_size = config['spectrogram']['overlap_size']
    
    # load waveform
    waveform, sr = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)
    samples_per_window = window_size * sr
    samples_overlap = overlap_size * sr
    wav_start = WAVhdr(audio_file_path).start

    # break wav into chunks
    chunks = []
    chunk_start_times = []
    chunk_end_times = []
    for start in range(0, waveform.shape[1], samples_per_window - samples_overlap):

        chunk_start_times.append(wav_start + timedelta(seconds = start / sr))
        end = start + samples_per_window  
        if end > waveform.shape[1]: # if the last chunk is smaller than the window size, pad it with zeros
            y_pad = torch.nn.functional.pad(waveform[:, start:], (0, end - waveform.shape[1]), mode='constant')
            chunks.append(y_pad)
            chunk_end_times.append(chunk_start_times[-1] + timedelta(seconds = (waveform.shape[1] - start) / sr))
        else:
            chunks.append(waveform[:, start:end])
            chunk_end_times.append(chunk_start_times[-1] + timedelta(seconds = samples_per_window / sr))
    
    return chunks, chunk_start_times, chunk_end_times, sr


def chunk_audio_xwav(audio_file_path, device):

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    window_size = config['spectrogram']['window_size']
    overlap_size = config['spectrogram']['overlap_size']

    # load waveform
    waveform, _ = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)
    xwav = XWAVhdr(audio_file_path)
    sr = xwav.xhd['SampleRate']
    n_channels = xwav.xhd['NumChannels']
    bytes_per_sample = xwav.xhd['BitsPerSample'] / 8
    
    # compute number of samples per raw file, number of samples per window, number of samples of overlap
    samples_per_raw = [int(b / (n_channels * bytes_per_sample)) for b in xwav.xhd['byte_length']]
    dnum_start = xwav.raw['dnumStart'] # datetime list of raw file starts
    samples_per_window = window_size * sr
    samples_overlap = overlap_size * sr

    # break xwav into chunks
    chunks = []
    chunk_start_times = []
    chunk_end_times = []
    buffer = torch.empty((waveform.shape[0], 0), dtype=waveform.dtype).to(device)
    current_time = None
    sample_ptr = 0
    prev_end_time = None
    
    for i, n_samples in enumerate(samples_per_raw):
        raw_time = dnum_start[i]
        raw_audio = waveform[:, sample_ptr:sample_ptr + n_samples]
        sample_ptr += n_samples
        # If there's a time gap > 1s, flush and pad the current buffer
        if prev_end_time != None and (raw_time - prev_end_time).total_seconds() > 1:
            if buffer.shape[1] > 0:
                padded = torch.nn.functional.pad(buffer, (0, samples_per_window - buffer.shape[1]), 'constant', 0)
                chunks.append(padded)
                chunk_start_times.append(current_time)
                chunk_end_times.append(prev_end_time)
            buffer = raw_audio
            current_time = raw_time
        # No gap → concatenate new audio onto buffer
        else:
            if buffer.shape[1] == 0:
                current_time = raw_time
            buffer = torch.cat((buffer, raw_audio), dim=1)
        # While buffer contains enough samples for one chunk → extract chunks
        while buffer.shape[1] >= samples_per_window:
            chunks.append(buffer[:, :samples_per_window])
            chunk_start_times.append(current_time)
            chunk_end_times.append(current_time + timedelta(seconds = samples_per_window / sr))
            buffer = buffer[:, samples_per_window - samples_overlap:]
            current_time += timedelta(seconds=window_size - overlap_size)
    
        prev_end_time = raw_time + timedelta(seconds=n_samples / sr)
        
    # pad final chunk if needed
    if buffer.shape[1] > 0:
        padded = torch.nn.functional.pad(buffer, (0, samples_per_window - buffer.shape[1]), 'constant', 0)
        chunks.append(padded)
        chunk_start_times.append(current_time)
        chunk_end_times.append(prev_end_time)
    
    return chunks, chunk_start_times, chunk_end_times, sr

def chunk_to_spectrogram(chunks, sr, device):
    '''
    Converts chunks of audio to spectrograms. These spectrograms are stored as tensors on the current torch device.
    Note that if the chunk contains multiple channels, only the first channel will be used.
    Inputs:
    - chunks: a list of audio chunks (tensors of samples), each of shape [C, N] for C channnels and N samples
    - sr: sample rate
    - device: the torch device to store data on
    Outputs:
    - spectrograms: a list of spectrograms, each stored as a tensor of type uint8 with range normalized to [0, 255]
    '''
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CalCOFI_flag = config['spectrogram']['CalCOFI_flag']
    min_freq = config['spectrogram']['min_freq']
    max_freq = config['spectrogram']['max_freq']
    sec_per_bin = config['spectrogram']['sec_per_bin']
    Hz_per_bin = config['spectrogram']['Hz_per_bin']
    
    spectrograms = []
    
    for chunk in chunks:
       
        S = torch.stft(
            chunk[0].to(device), 
            n_fft=int(sr / Hz_per_bin), # Hz_per_bin Hz per frequency bin
            hop_length=int(sr * sec_per_bin), # sec_per_bin seconds per time bin
            window=torch.hamming_window(int(sr / Hz_per_bin)).to(device), 
            return_complex=True
        ).cpu()  
        
        transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) # convert to dB and clip at 80dB
        S_db = transform(torch.abs(S))[int(min_freq / Hz_per_bin):int(max_freq / Hz_per_bin) + 1, :] # set freuency range
        S_db_normalized = (S_db - torch.min(S_db)) / (torch.max(S_db) - torch.min(S_db))          
        S_img = (255 * torch.flip(S_db_normalized, dims=[0])).to(torch.uint8) # convert to image format
        
        if CalCOFI_flag:
            S_img = remove_AIS(S_img)
        spectrograms.append(S_img)

    return spectrograms


def name_spectrogram_file(wav_file_name, chunk_start_time):
    '''
    Defines file naming convention for saved spectrograms
    '''
    timestamp_str = chunk_start_time.strftime('%Y%m%dT%H%M%S')
    return f'{wav_file_name}_{timestamp_str}.png'

    
def remove_AIS(spectrogram):
    '''
    Function to mask out AIS nagivational signals in spectrograms. Detected signals are covered with gray overlays.
    '''
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    Hz_per_bin = config['spectrogram']['Hz_per_bin']
    thresholds = [2000, 1800, 1600]  # Threshold values for detecting AIS signal in the first, second, and third 10 Hz blocks 
    thresholds = [int(threshold / Hz_per_bin) for threshold in thresholds] # scaling thresholds based on frequency resolution
    gray_value = 128  # Gray value to replace the AIS signal
    
    # Find the vertical white lines and gray them out
    for col in range(spectrogram.shape[1]):
        if (torch.sum(spectrogram[int(-10 / Hz_per_bin):, col]) > thresholds[0] and 
            torch.sum(spectrogram[int(-20 / Hz_per_bin):int(-10 / Hz_per_bin), col]) > thresholds[1] and 
            torch.sum(spectrogram[int(-30 / Hz_per_bin):int(-20 / Hz_per_bin), col]) > thresholds[2]):         
              spectrogram[:, col] = gray_value  # Replace the entire column with gray 
    
    return spectrogram


def freq_to_pixel(freq):
    """
    Map frequency to pixel position in the spectrogram
    """

    # load spectrogram parameters
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    min_freq = config['spectrogram']['min_freq']
    max_freq = config['spectrogram']['max_freq']
    Hz_per_bin = config['spectrogram']['Hz_per_bin']
    freq_range = max_freq - min_freq + 1
    num_freq_bins = int(freq_range / Hz_per_bin)

    # calculate pixel position, pixel position is calculated from the bottom (low frequencies) because of the inversion
    y = round(num_freq_bins - 1 - (freq - min_freq) / Hz_per_bin)

    # ensure y is within spectrogram bounds
    y = max(0, y)
    y = min(num_freq_bins - 1, y)
    
    return y

def pixel_to_freq(y):
    """
    Map pixel to frequency position in the spectrogram
    """

    # load spectrogram parameters
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    min_freq = config['spectrogram']['min_freq']
    max_freq = config['spectrogram']['max_freq']
    Hz_per_bin = config['spectrogram']['Hz_per_bin']

    # calculate frequency, pixel position is calculated from the bottom (low frequencies) because of the inversion
    freq = round(max_freq - y * Hz_per_bin)

    # ensure frequency is within spectrogram bounds
    freq = max(min_freq, freq)
    freq = min(max_freq, freq)
    
    return freq


def time_to_pixel(time):
    """
    Map time to pixel position in the spectrogram
    """

    # load spectrogram parameters
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    sec_per_bin = config['spectrogram']['sec_per_bin']
    window_size = config['spectrogram']['window_size']
    num_time_bins = int(window_size / sec_per_bin)

    # calculate pixel position
    x = round(time / sec_per_bin)
    
    # ensure x is within spectrogram bounds
    x = max(0, x)
    x = min(num_time_bins - 1, x)
    
    return x


def pixel_to_time(x):
    """
    Map pixel to time position in the spectrogram
    NOTE: This function does not truncate as the other conversion functions do. This is to preserve time resolution
    """

    # load spectrogram parameters
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    sec_per_bin = config['spectrogram']['sec_per_bin']
    window_size = config['spectrogram']['window_size']

    # calculate pixel position
    time = x * sec_per_bin
    
    # ensure time is within spectrogram bounds
    time = max(0, time)
    time = min(window_size - sec_per_bin, time)
    
    return time