import numpy as np
import gdown

from . import FastPitch2Wave, files_dict, _VOWELIZER
from ..utils.audio import save_wave
from pathlib import Path
try:
    import sounddevice as sd
except:
    pass

def play_wave(wave, 
              sr: int = 22050, 
              blocking: bool = False
              ) -> None:
    sd.play(wave, samplerate=sr, blocking=blocking)



def get_model_path(package_path: Path = None, 
                   name: str = "fastpitch"
                   ) -> str:
    file_entry = files_dict[name]
    if package_path is None:
        package_path = Path(__file__).parent.parent
    model_path = package_path.joinpath(file_entry['file']) 
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    if not model_path.exists() or (model_path.lstat().st_mtime < file_entry.get('timestamp', 0)):
        gdown.download(file_entry['url'], output=model_path.as_posix(), fuzzy=True)
    
    return model_path.as_posix()

def get_model(name: str = 'fastpitch2wave', 
              cuda: bool =True
              ) -> FastPitch2Wave:   
    package_path = Path(__file__).parent.parent

    fastpitch_path = get_model_path(package_path, 'fastpitch')
    hifigan_path = get_model_path(package_path, 'hifigan')
    denoiser_path = get_model_path(package_path, 'denoiser')
    tts_model = FastPitch2Wave(
        fastpitch_path, hifigan_path,
        denoiser_path, cuda=cuda)
    
    return tts_model

def tts(text: str, 
        speaker: int = 0,
        pace: float = 1.,
        denoise: float = 0.005,   
        play: bool = False,
        vowelizer: _VOWELIZER = None,
        pitch_mul: float = 1.,
        pitch_add: float = 0.,      
        save_to: str = None,
        bits_per_sample: int = 32,
        cuda: bool = True
        ) -> np.ndarray:
    """
    Parameters:
        text (str): Text
        speaker (int): Speaker id (0-3)
        pace (float): Speaker pace
        denoise (float): Denoiser strength
        play (bool): Play audio? 
        vowelizer [shakkala|shakkelha]: Optional; Vowelizer model
        save_to (str): Optional; Filepath where audio WAV file is saved 
        bits_per_sample (int): when `save_to` is specified (8, 16 or 32 bit)
        cuda (bool): Use CUDA provider?
        
    Returns:
        (ndarray): Waveform sampled at 22050Hz, shape: [n_samples]
        
    Examples:
        >>> from tts_arabic import tts
        >>> text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
        >>> wave = tts(text, speaker=2, pace=0.9, play=True)
        # Buckwalter transliteration
        >>> text = ">als~alAmu Ealaykum yA Sadiyqiy."
        >>> wave = tts(text, speaker=0, play=True)   
        # Unvocalized input
        >>> text_unvoc = "القهوة مشروب يعد من بذور البن المحمصة."
        >>> wave = tts(text_unvoc, play=True, vowelizer='shakkelha')
        
    """
    if not hasattr(tts, 'model'):
        setattr(tts, 'model', get_model(cuda=cuda))
    
    wave_out = tts.model.infer(text, speaker, pace, denoise, 
                               vowelizer=vowelizer, 
                               pitch_mul=pitch_mul,
                               pitch_add=pitch_add,)
    if play: play_wave(wave_out)
    if save_to is not None:
        save_wave(wave_out, save_to, sample_rate=22050, 
                  bits_per_sample=bits_per_sample)
        
    
    return wave_out
