ARABIC_LETTERS_LIST = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي'

DIACRITICS_LIST = ['َ', 'ً', 'ِ', 'ٍ', 'ُ', 'ٌ', 'ْ', 'ّ']

def remove_diacritics(data):
  return data.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))