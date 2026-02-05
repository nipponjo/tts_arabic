
import numpy as np
import onnxruntime as ort
from .tashkeel_tokenizer_mod import TashkeelTokenizer
from .utils import remove_non_arabic


class CATTModel:
    def __init__(self, 
                 sd_path: str = "data/catt_eo.onnx",      
                 cuda: bool = None) -> None:
        providers = ['CPUExecutionProvider']
        if cuda is not None:           
            if not isinstance(cuda, int): cuda = 0
            providers.insert(0, ('CUDAExecutionProvider', {
                'device_id': cuda
            }))
        self.ort_sess = ort.InferenceSession(
            sd_path, providers=providers)
        
        self.tokenizer = TashkeelTokenizer()

    def predict(self, text: str, return_probs: bool=False
              ) -> np.ndarray:
        """
        Parameters:
            text (str): Text
        
        Returns:
            (str): Diacritized text
        """
        
        
        text = remove_non_arabic(text)

        input_ids, _ = self.tokenizer.encode(text, test_match=False)        
        input_ids = np.array(input_ids, dtype=np.int64)[None][:, 1:-1]
        
        y_pred_probs = self.ort_sess.run(None, {'in_token_ids': input_ids})[0]        
        y_pred = y_pred_probs.argmax(-1)
        
        y_pred[self.tokenizer.letters_map[' '] == input_ids] = self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]
        text_with_tashkeel = self.tokenizer.decode(input_ids, y_pred)[0]
        
        if return_probs:
            return text_with_tashkeel, y_pred_probs
        
        return text_with_tashkeel
    