'''
Functions to work with XWAV files
Author: Joshua Zingale
Time: June 3rd, 2024
'''
from AudioStreamDescriptor import XWAVhdr
from datetime import timedelta

def get_datetime(xwav_time: float, xwav):
    '''Given the time in seconds from the beginning of an XWAV along with
    the XWAV, returns the corresponding absolute datetime'''
    
    # The length of each block in seconds
    BLOCK_LEN = 75

    # Allow xwav to be a string or XWAVhdr
    if type(xwav) != XWAVhdr:
        xwav = XWAVhdr(xwav)

    # Most recent block for the starttime
    block_offset = xwav_time % BLOCK_LEN # calculate remainder
    block_idx = int(xwav_time // BLOCK_LEN)

    return xwav.raw['dnumStart'][block_idx] + timedelta(seconds = block_offset)


