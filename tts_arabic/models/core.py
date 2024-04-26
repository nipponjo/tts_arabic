import numpy as np
import gdown
from pathlib import Path

from . import FastPitch2Wave, files_dict, _VOWELIZER

_MODEL_TYPE = str

try:
    import sounddevice as sd
except:
    pass

def play_wave(wave, 
              sr: int = 22050, 
              blocking: bool = False
              ) -> None:
    sd.play(wave, samplerate=sr, blocking=blocking)

def get_model_path(package_path, name="fastpitch"):
    model_path = package_path.joinpath(files_dict[name]['file'])     
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    if not model_path.exists():
        gdown.download(files_dict[name]['url'], output=model_path.as_posix(), fuzzy=True)
    return model_path.as_posix()

def get_model(name='fastpitch2wave'):   
    package_path = Path(__file__).parent.parent

    fastpitch_path = get_model_path(package_path, 'fastpitch')
    hifigan_path = get_model_path(package_path, 'hifigan')
    denoiser_path = get_model_path(package_path, 'denoiser')
    tts_model = FastPitch2Wave(
        fastpitch_path, hifigan_path,
        denoiser_path)
    
    return tts_model

def tts(text: str, 
        speaker: int = 0,
        pace: float = 1.,
        denoise: float = 0.005,   
        play: bool = False,
        vowelizer: _VOWELIZER = None,
        ) -> np.ndarray:
    """
    Parameters:
        text (str): Text
        speaker (int): Speaker id (0-3)
        pace (float): Speaker pace
        denoise (float): Denoiser strength   
        vowelizer [shakkala|shakkelha]: Vowelizer model 
        
    Returns:
        (ndarray): Waveform sampled at 22050Hz, shape: [n_samples]
        
    Examples:
        >>> from tts_arabic import tts
        >>> text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
        >>> wave = tts(text, pace=0.9, speaker=2, play=True)
        # Buckwalter transliteration
        >>> text = ">als~alAmu Ealaykum yA Sadiyqiy."
        >>> wave = tts(text, speaker=0, play=True)   
        # Unvocalized input
        >>> text_unvoc = "القهوة مشروب يعد من بذور البن المحمصة."
        >>> wave = tts(text_unvoc, play=True, vowelizer='shakkelha')
    """
    if not hasattr(tts, 'model'):
        setattr(tts, 'model', get_model())
    
    wave_out = tts.model.infer(text, speaker, pace, denoise, 
                               vowelizer=vowelizer)
    if play: play_wave(wave_out)
    
    return wave_out
