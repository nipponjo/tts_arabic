Arabic TTS model (FastPitch) from the [tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch) repo in the ONNX format.

Audio samples can be found [here](https://nipponjo.github.io/tts-arabic-speakers).

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

TTS options:
```python
from tts_arabic import tts

text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
wave = tts(
    text, # input text
    speaker = 1, # speaker id; choose between 0,1,2,3
    pace = 1, # speaker pace
    denoise = 0.005, # HiFiGAN denoiser strength
    pitch_mul = 1, # pitch multiplier
    pitch_add = 0, # pitch offset
    play = True, # play audio?
    save_to = './test.wav', # Optionally; save audio WAV file
    bits_per_sample = 32, # when save_to is specified (8, 16 or 32 bits)
    vowelizer = None, # vowelizer model
    cuda = True, # use CUDA provider?
    )

```
