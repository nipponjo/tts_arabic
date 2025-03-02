from typing import Literal, Optional, get_args
import numpy as np
import gdown

from . import FastPitch2Wave, files_dict, _VOWELIZER
from ..utils.audio import save_wave
from pathlib import Path
try:
    import sounddevice as sd
except:
    pass

_MODEL_ID = Literal['fastpitch', 'mixer128', 'mixer80']
_VOCODER_ID = Literal['hifigan', 'vocos']


def play_wave(wave, 
              sr: int = 22050, 
              blocking: bool = False
              ) -> None:
    sd.play(wave, samplerate=sr, blocking=blocking)



def get_model_path(package_path: Optional[Path] = None, 
                   name: str = "fastpitch"
                   ) -> str:
    file_entry = files_dict[name]
    if package_path is None:
        package_path = Path(__file__).parent.parent
    model_path = package_path.joinpath(file_entry['file']) 
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    if not model_path.exists(): #or (model_path.lstat().st_mtime < file_entry.get('timestamp', 0)):
        gdown.download(file_entry['url'], output=model_path.as_posix(), fuzzy=True)
    
    return model_path.as_posix()


def get_model(model_id: _MODEL_ID = 'fastpitch',
              vocoder_id: _VOCODER_ID = 'hifigan',
              cuda: bool = True
              ) -> FastPitch2Wave:   
    package_path = Path(__file__).parent.parent

    fastpitch_path = get_model_path(package_path, model_id)
    hifigan_path = get_model_path(package_path, vocoder_id)    
    denoiser_path = get_model_path(package_path, 'denoiser') \
        if vocoder_id == 'hifigan' else None
    
    tts_model = FastPitch2Wave(
        fastpitch_path, 
        hifigan_path,
        denoiser_path,
        vocoder_id=vocoder_id,
        cuda=cuda)
    
    return tts_model


def get_available_models():
    return {
        'models': get_args(_MODEL_ID),
        'vocoders': get_args(_VOCODER_ID),
    }


def tts(text: str, 
        speaker: int = 0,
        pace: float = 1.,
        denoise: float = 0.005,   
        play: bool = False,
        vowelizer: Optional[_VOWELIZER] = None,
        pitch_mul: float = 1.,
        pitch_add: float = 0., 
        cuda: Optional[int] = None,
        model_id: _MODEL_ID = 'fastpitch',
        vocoder_id: _VOCODER_ID = 'hifigan',
        save_to: Optional[str] = None,
        bits_per_sample: int = 32,        
        ) -> np.ndarray:
    """
    Parameters:
        text (str): Text
        speaker (int): Speaker ID (0-3)
        pace (float): Speaker pace
        denoise (float): Denoiser strength        
        play (bool): Play audio?
        vowelizer [shakkala|shakkelha]: Optional; Vowelizer model
        pitch_mul (float): Pitch multiplier
        pitch_add (float): Pitch offset
        cuda (int): Optional; CUDA device index
        model_id (str): Model ID for Text->Mel model
        vocoder_id (str): Model ID for vocoder model
        save_to (str): Optional; Filepath where audio WAV file is saved 
        bits_per_sample (int): when `save_to` is specified (8, 16 or 32 bit)
        
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
    model_params = (model_id, vocoder_id, cuda)
    if not hasattr(tts, 'model') or tts.params != model_params:
        setattr(tts, 'model', get_model(*model_params))
        setattr(tts, 'params', model_params)    
    
    wave_out = tts.model.infer(text, speaker, pace, denoise, 
                               vowelizer=vowelizer, 
                               pitch_mul=pitch_mul,
                               pitch_add=pitch_add,)
    
    if play: play_wave(wave_out)
    if save_to is not None:
        save_wave(wave_out, save_to, sample_rate=22050, 
                  bits_per_sample=bits_per_sample)
        
    
    return wave_out
