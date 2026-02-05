"""
Arabic text-to-speech (TTS) with neural FastPitch and MixerTTS models in ONNX format.

This package provides a lightweight, inference-only Arabic TTS pipeline supporting
Modern Standard Arabic (MSA) with optional automatic diacritization. Models are
exported to ONNX for fast CPU and GPU inference and are loaded lazily on first use.

Key features
------------
- Neural Text→Mel synthesis (FastPitch, MixerTTS)
- Neural vocoding via HiFi-GAN or Vocos
- Support for:
  * Fully diacritized Arabic
  * Undiacritized Arabic with optional vowelization
  * Buckwalter transliteration
- Multi-speaker synthesis (model-dependent)
- Pitch, pace, volume, and denoising control
- Optional real-time audio playback
- WAV file export with configurable bit depth

Public API
----------
tts
    Synthesize speech from Arabic text and return a waveform.
get_available_models
    List available Text→Mel and vocoder model identifiers.
get_model
    Load a TTS model/vocoder pair instance.
play_wave
    Play a waveform array using the system audio device.
save_wave
    Save a waveform array to a WAV file.
vocalize
    Apply automatic Arabic diacritization to unvocalized text.

Examples
--------
>>> from tts_arabic import tts, get_available_models
>>> print(get_available_models())

Diacritized Arabic input:
>>> text = "السَّلامُ عَلَيكُم يَا صَدِيقِي."
>>> wave = tts(text, speaker=2, pace=0.9, play=True)

Buckwalter transliteration:
>>> text = ">als~alAmu Ealaykum yA Sadiyqiy."
>>> wave = tts(text, speaker=0, play=True)

Undiacritized input with automatic vowelization:
>>> text = "القهوة مشروب يعد من بذور البن المحمصة"
>>> wave = tts(text, play=True, vowelizer="shakkelha")
"""
from .models.core import (
    tts, 
    play_wave, 
    get_model, 
    save_wave, 
    get_available_models
    )
from .vocalizer.models.core import vocalize