import numpy as np
import onnxruntime as ort

from . import text as text_utils, _VOWELIZER
from ..vocalizer.models.core import vocalize

class FastPitch2Mel:
    def __init__(self, 
                 sd_path: str = "data/fp_ms.onnx",
                 arabic_in: bool = True) -> None:        
        self.ort_sess = ort.InferenceSession(
            sd_path, 
            providers=['CUDAExecutionProvider', 
                       'CPUExecutionProvider'])
        self.arabic_in = arabic_in
        
    def _vowelize(self,
                  utterance: str, 
                  vowelizer: _VOWELIZER = None
                  ) -> str:  
        if vowelizer is None: 
            return utterance
        utterance_ar = text_utils.buckwalter_to_arabic(utterance)
        return vocalize(utterance_ar, model=vowelizer)
        
    def _tokenize(self, 
                  utterance: str, 
                  vowelizer: _VOWELIZER = None
                  ) -> list[str]:    
        utterance = self._vowelize(utterance=utterance, vowelizer=vowelizer)
        if self.arabic_in:
            return text_utils.arabic_to_tokens(utterance)
        return text_utils.buckwalter_to_tokens(utterance)

    def infer(self, 
              text: str, 
              pace: float = 1., 
              speaker: int = 0,
              vowelizer: _VOWELIZER = None,
              ) -> np.ndarray:
        """
        Parameters:
            text (str): Text
            pace (float): Speaker pace
            speaker (int): Speaker id
        
        Returns:
            (ndarray): Mel spectrogram, shape: [mel_bands, n_frames]
        """
        tokens = self._tokenize(text, vowelizer=vowelizer)
        token_ids = text_utils.tokens_to_ids(tokens)
        ids_batch = np.array(token_ids, dtype=np.int64)[None]
        mel_spec = self.ort_sess.run(
            None, {
                "input": ids_batch, 
                "pace": np.array([pace], dtype=np.float64),
                "speaker": np.array([speaker], dtype=np.int64),
            },)[0].astype(np.float32)
        
        return mel_spec[0]
    

class HifiGanVocoder:
    def __init__(self, 
                 sd_path: str = "data/hifigan.onnx") -> None:
        self.ort_sess = ort.InferenceSession(
            sd_path, 
            providers=['CUDAExecutionProvider', 
                       'CPUExecutionProvider'])
    
    def infer(self, 
              mel_spec: np.ndarray
              ) -> np.ndarray:
        """
        Parameters:
            mel_spec (ndarray): Mel spectrogram, shape: [mel_bands, n_frames]
        
        Returns:
            (ndarray): Waveform, shape: [n_samples]
        """
        if mel_spec.ndim == 2:
            mel_spec = mel_spec[None]
        wave_out = self.ort_sess.run(
            None,
            {"input": mel_spec},
        )[0].astype(float)
        
        return wave_out[0, 0]
    

class FastPitch2Wave:
    def __init__(self,
                 sd_path_ttmel: str = "data/fp_ms.onnx",
                 sd_path_mel2wave: str = "data/hifigan.onnx"
                 ) -> None:
        self.ttmel_model = FastPitch2Mel(sd_path_ttmel)
        self.mel2wave_model = HifiGanVocoder(sd_path_mel2wave)
    
    def infer(self, 
              text: str, 
              pace: float = 1., 
              speaker: int = 0,
              vowelizer: _VOWELIZER = None,        
              ) -> np.ndarray:
        """
        Parameters:
            text (str): Text
            pace (float): Speaker pace
            speaker (int): Speaker id
            
        Returns:
            (ndarray): Waveform sampled at 22050Hz, shape: [n_samples]
        """
        mel_spec = self.ttmel_model.infer(text, 
                                          pace=pace, 
                                          speaker=speaker,
                                          vowelizer=vowelizer)
        wave_out = self.mel2wave_model.infer(mel_spec)
        
        return wave_out