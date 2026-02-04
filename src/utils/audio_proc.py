import os
import torch
import numpy as np
import torch.nn.functional as F

def trim_or_pad_audio(audio, t=1.0, fs=16000):
    '''
    audio (Tensor): Tensor of audio of dimension (Time)
    '''
    max_len = int(t*fs)
    shape = audio.shape
    if shape[0] >= max_len:
        audio = audio[:max_len]
    else:
        n_pad = max_len - shape[0]
        zero_shape = (n_pad,)
        audio = torch.cat((audio, torch.zeros(zero_shape)), axis=0)
    return audio
