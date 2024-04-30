from pathlib import Path
from typing import Literal, List, Union
import gdown

from ...urls import files_dict
from ..models import Shakkala, Shakkelha

_MODEL_TYPE = Literal['shakkala', 'shakkelha']


def get_model_path(package_path, name="fastpitch") -> str:
    model_path = package_path.joinpath(files_dict[name]['file'])     
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    if not model_path.exists():
        gdown.download(files_dict[name]['url'], output=model_path.as_posix(), fuzzy=True)
    return model_path.as_posix()


def get_model(model: _MODEL_TYPE = 'shakkelha'):
    assert model in ('shakkala', 'shakkelha')

    # data_folder = Path(__file__).parent.parent.parent.joinpath('data')
    package_path = Path(__file__).parent.parent.parent
    model_path = get_model_path(package_path, model)  
    if model == 'shakkala':      
        return Shakkala(sd_path=model_path)     
    elif model == 'shakkelha':
        return Shakkelha(sd_path=model_path) 


def vocalize(input_text: Union[str, List[str]], 
             model: _MODEL_TYPE = 'shakkelha',
             return_probs: bool = False
             ) -> Union[str, List[str]]:
    """
    Parameters:
        input_text (str|list[str]): Unvocalized text
        model: Vocalization model [shakkala|shakkelha]
        return_probs: Return probabilities?
        
    Returns:
        (str|list[str]): Predicted vocalized text
    
    Examples:
        >>> input_text = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم، يتحدثها أكثر من 467 مليون نسمة"
        # shakkala output
        >>> print(vocalize(input_text, model='shakkala'))
        >>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مَنْ 467 مُلْيُونُ نُسْمَةَ
        # shakkelha output
        >>> print(vocalize(input_text, model='shakkelha'))
        >>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مِنْ 467 مَلْيُونٍ نَسَمَةً

    """
    assert model in ('shakkala', 'shakkelha')
    if not hasattr(vocalize, model):
        setattr(vocalize, model, get_model(model=model))

    if model == 'shakkala':
        return vocalize.shakkala.predict(input_text, return_probs=return_probs)
    elif model == 'shakkelha':
        return vocalize.shakkelha.predict(input_text, return_probs=return_probs)
    else:
        return  