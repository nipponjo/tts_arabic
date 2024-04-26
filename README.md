Arabic TTS model (FastPitch) from the [tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch) repo in the ONNX format.

Install with:
```
pip install git+https://github.com/nipponjo/tts_arabic.git
```

Examples:
```python

# %%
from tts_arabic import tts

# %%
text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
wave = tts(text, speaker=2, pace=0.9, play=True)

# %% Buckwalter transliteration
text = ">als~alAmu Ealaykum yA Sadiyqiy."
wave = tts(text, speaker=0, play=True)

# %% Unvocalized input
text_unvoc = "القهوة مشروب يعد من بذور البن المحمصة"
wave = tts(text_unvoc, play=True, vowelizer='shakkelha')


```

## TODO

- [ ] Add batch support
- [ ] Extend Readme
