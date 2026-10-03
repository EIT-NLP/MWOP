from typing import List
import numpy as np
from librosa import resample

def downsample_audio(audio_array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    audio_resample_array = resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resample_array

def split_audio(audio_arrays: np.ndarray, chunk_lim: int) -> List:
    audio_splits = []
    for i in range(0, len(audio_arrays), chunk_lim):
        audio_splits.append(audio_arrays[i:i + chunk_lim])
    return audio_splits
