# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:09:05 2024

@author: Michaela Alksne
"""
from AudioStreamDescriptor import WAVhdr

# hepler function uses WAVhdr to read wav file header info and extract wav file start time as a datetime object

def extract_wav_start(path):
    wav_hdr = WAVhdr(path)
    wav_start_time = wav_hdr.start
    return wav_start_time


