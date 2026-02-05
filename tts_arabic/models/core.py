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
_VOCODER_ID = Literal['hifigan', 'vocos', 'vocos44']
_vocoder_id_to_sr = {
    'hifigan': 22050, 'vocos': 22050, 'vocos44': 44100,
}


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
    # or (model_path.lstat().st_mtime < file_entry.get('timestamp', 0)):
    if not model_path.exists():
        gdown.download(file_entry['url'],
                       output=model_path.as_posix(), fuzzy=True)

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
        volume: float = 0.9,
        play: bool = False,
        vowelizer: Optional[_VOWELIZER] = None,
        pitch_mul: float = 1.,
        pitch_add: float = 0.,
        cuda: Optional[int] = None,
        model_id: _MODEL_ID = 'fastpitch',
        vocoder_id: _VOCODER_ID = 'hifigan',
        save_to: Optional[str] = None,
        bits_per_sample: int = 32,
        return_mel: bool = False,
        blocking: bool = True,
        ) -> np.ndarray:
    """
    Synthesize speech from Arabic text using a neural TTS pipeline.

    This function performs text-to-speech (TTS) synthesis by converting input
    text into a waveform via a Text→Mel model and a neural vocoder. It supports
    vocalized (diacritized) Arabic, unvocalized Arabic with optional automatic
    diacritization, and Buckwalter transliteration.

    Models are lazily loaded and cached per (model_id, vocoder_id, cuda) tuple.

    Parameters
    ----------
    text : str
        Input text to synthesize. Can be:
        - Fully diacritized Arabic
        - Undiacritized Arabic (optionally processed via `vowelizer`)
        - Buckwalter transliteration

    speaker : int, default=0
        Speaker ID. Valid range depends on the selected model
        (typically 0-3 for multi-speaker models).

    pace : float, default=1.0
        Speaking rate multiplier.
        Values < 1.0 slow down speech; values > 1.0 speed it up.

    denoise : float, default=0.005
        Post-vocoder bias correction strength.
        This subtracts a learned spectral bias from the generated audio.
        Low values clean up subtle vocoder artifacts; high values suppress
        high frequencies and reduce brightness, potentially making speech
        sound flat or muffled. Use sparingly (e.g. 0.002-0.02).

    volume : float, default=0.9
        Output waveform peak amplitude, clipped to [0, 1].

    play : bool, default=False
        If True, play the synthesized audio immediately.

    vowelizer : Optional[str], default=None
        Optional Arabic diacritizer applied to unvocalized input text
        before synthesis (e.g., "shakkelha" or "catt_eo").

    pitch_mul : float, default=1.0
        Multiplicative scaling factor applied to the normalized pitch contour
        (z-score normalized using the speaker’s global mean and standard deviation).
        Values > 1.0 amplify pitch variation around the speaker’s mean
        (more expressive intonation), while values between 0 and 1.0
        compress pitch variation (flatter intonation).
        Negative values invert pitch deviations around the mean, resulting
        in reversed intonation patterns.

    pitch_add : float, default=0.0
        Additive offset applied in normalized (z-score) pitch space.
        Shifts the entire pitch contour upward or downward relative to the
        speaker’s mean pitch without changing relative variation.
        A value of ±1.0 corresponds approximately to a ±1 standard deviation
        shift from the speaker’s average pitch.

    cuda : Optional[int], default=None
        CUDA device index to use. If None, runs on CPU.

    model_id : str, default="fastpitch"
        Identifier of the Text→Mel model.

    vocoder_id : str, default="hifigan"
        Identifier of the neural vocoder.

    save_to : Optional[str], default=None
        If provided, save the synthesized waveform as a WAV file
        at the given path.

    bits_per_sample : int, default=32
        Bit depth of the saved WAV file when `save_to` is specified.
        Supported values: 8, 16, or 32.

    return_mel : bool, default=False
        If True, return both the waveform and the intermediate mel-spectrogram
        as a tuple (waveform, mel).

    blocking : bool, default=True
        If `play=True`, block execution until playback finishes.

    Returns
    -------
    numpy.ndarray or tuple
        - Waveform array of shape (n_samples,) sampled at the vocoder’s
          native sample rate (typically 22050 Hz), or
        - (waveform, mel) if `return_mel=True`.

    Examples
    --------
    >>> from tts_arabic import tts
    >>> text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
    >>> wave = tts(text, speaker=2, pace=0.9, play=True)

    Buckwalter transliteration:
    >>> text = ">als~alAmu Ealaykum yA Sadiyqiy."
    >>> wave = tts(text, speaker=0, play=True)

    Undiacritized input with automatic vowelization:
    >>> text = "القهوة مشروب يعد من بذور البن المحمصة."
    >>> wave = tts(text, play=True, vowelizer="shakkelha")
    """
    model_params = (model_id, vocoder_id, cuda)
    if not hasattr(tts, 'model') or tts.params != model_params:
        # cache models
        setattr(tts, 'model', get_model(*model_params))
        setattr(tts, 'params', model_params)
        setattr(tts, 'sr', _vocoder_id_to_sr.get(vocoder_id, 22050))

    # TTS inference
    output = tts.model.infer(
        text, speaker, pace,
        denoise,
        volume=volume,
        vowelizer=vowelizer,
        pitch_mul=pitch_mul,
        pitch_add=pitch_add,
        return_mel=return_mel,
    )
    wave_out = output[0] if isinstance(output, tuple) else output
    if play:
        play_wave(wave_out, blocking=blocking, sr=tts.sr)
    if save_to is not None:
        save_wave(wave_out, save_to, sample_rate=tts.sr,
                  bits_per_sample=bits_per_sample)

    return output
