# from typing import Literal
# _VOWELIZER = Literal['catt', 'shakkala', 'shakkelha']

from .. import text
from .. import vocalizer
from ..vocalizer.models.core import _MODEL_TYPE as _VOWELIZER
from .tts_models import FastPitch2Wave, FastPitch2Mel, VocosVocoder, HifiGanVocoder
from ..urls import files_dict
