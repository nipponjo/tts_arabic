import wave
import numpy as np

from typing import Literal


def save_wave(audio_data: np.ndarray,
              audio_path: str,
              sample_rate: int = 22050,  # Sample rate in Hz
              bits_per_sample: Literal[8, 16, 24, 32] = 32,
              normalize: bool = False
              ) -> None:
    """
    Args:
        audio_data (ndarray): shape [n_samples]
        audio_path (str): 
        sample_rate (int): 
        bits_per_sample (int): 8, 16, 24 or 32 bits per sample
        normalize (bool):    

    """
    assert bits_per_sample in (8, 16, 24, 32)
    assert audio_data.ndim == 1

    # Define the parameters of the audio file
    channels = 1  # Mono audio

    sample_width = bits_per_sample // 8
    int_value_max = (256**sample_width) // 2 - 1

    audio_abs_max = np.abs(audio_data).max()
    if audio_abs_max > 1 or normalize:
        audio_data /= audio_abs_max

    # Normalize the audio data to fit within the range of the chosen bit depth
    audio_data = audio_data * int_value_max

    if bits_per_sample == 24:
        audio_data = audio_data.astype(np.int32)
        audio_bytes = audio_data.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        audio_type = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        audio_bytes = audio_data.astype(audio_type).tobytes()

    # Create a new wave file
    with wave.open(audio_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
