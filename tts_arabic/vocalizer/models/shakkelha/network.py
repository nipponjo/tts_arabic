import numpy as np
import onnxruntime as ort
from typing import Union, List

from . import encode, decode

class Shakkelha:
    def __init__(self, sd_path: str=None):     
        self.ort_sess = ort.InferenceSession(
            sd_path, providers=['CUDAExecutionProvider', 
                                'CPUExecutionProvider'])   
        

    def infer(self, x):
        return self.ort_sess.run(
            None, {"input": x},)[0] 
    
    def _predict_list(self, input_list: List[str], return_probs: bool=False):
        output_list = []
        probs_list = []
        for input_text in input_list:
            if return_probs:
                output_text, probs = self._predict_single(input_text, return_probs=True)
                output_list.append(output_text)
                probs_list.append(probs[0, 1:-1])
            else:
                output_list.append(self._predict_single(input_text))

        if return_probs:
            return output_list, probs_list
        
        return output_list
    
    def _predict_single(self, input_text: str, return_probs: bool=False):
        ids = encode(input_text)
        input = np.array(ids, dtype=np.int64)[None]
        probs = self.infer(input)
        output = decode(probs, input_text)

        if return_probs:
            return output, probs[0, 1:-1]
        
        return output

    def predict(self, input: Union[str, List[str]], return_probs: bool=False):
        if isinstance(input, str):
            return self._predict_single(input, return_probs=return_probs)        
        
        return self._predict_list(input, return_probs=return_probs)