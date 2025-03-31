Arabic TTS model (FastPitch, MixerTTS) from the [tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch) repo in the ONNX format.

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

Pretrained models:
|Model|Model ID|Type|#params|Paper|Output|
|-------|---|---|------|----|----|
|FastPitch|fastpitch|Text->Mel|46.3M|[arxiv](https://arxiv.org/abs/2006.06873)|Mel (80 bins)|
|MixerTTS|mixer128|Text->Mel|2.9M|[arxiv](https://arxiv.org/abs/2110.03584)|Mel (80 bins)|
|MixerTTS|mixer80|Text->Mel|1.5M|[arxiv](https://arxiv.org/abs/2110.03584)|Mel (80 bins)|
|HiFi-GAN|hifigan|Vocoder|13.9M|[arxiv](https://arxiv.org/abs/2010.05646)|Wave (22.05kHz)|
|Vocos|vocos|Vocoder|13.4M|[arxiv](https://arxiv.org/abs/2306.00814)|Wave (22.05kHz)|
|Vocos|vocos44|Vocoder|14.0M|[arxiv](https://arxiv.org/abs/2306.00814)|Wave (44.1kHz)|

The sequence of transformations is as follows:

*Text* &rarr; Phonemizer &rarr; *Phonemes* &rarr; Tokenizer &rarr; *Token Ids* &rarr; **Text->Mel** model &rarr; *Mel spectrogram* &rarr; **Vocoder** model &rarr; *Wave*

The `Text->Mel` models map token ids to mel frames. All models use the 80 bin configuration proposed by [HiFi-GAN](https://github.com/jik876/hifi-gan). This mel spectrogram contains frequencies up to 8kHz. The `vocoder` models map the mel spectrogram to a waveform. The vocoders with `vocoder_id` `hifigan` and `vocos` artificially extend the bandwidth to 11025Hz, and `vocos44` to 22050Hz. Samples for comparing the models can be found [here](https://nipponjo.github.io/tts-arabic-speakers/#models-cmp).

TTS options:
```python
from tts_arabic import tts

text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
wave = tts(
    text, # input text
    speaker = 1, # speaker id; choose between 0,1,2,3
    pace = 1, # speaker pace
    denoise = 0.005, # vocoder denoiser strength
    volume = 0.9, # Max amplitude (between 0 and 1)
    play = True, # play audio?
    pitch_mul = 1, # pitch multiplier
    pitch_add = 0, # pitch offset
    vowelizer = None, # vowelizer model
    model_id = 'fastpitch', # Model ID for Text->Mel model
    vocoder_id = 'hifigan', # Model ID for vocoder model
    cuda = None, # Optional; CUDA device index
    save_to = './test.wav', # Optionally; save audio WAV file
    bits_per_sample = 32, # when save_to is specified (8, 16 or 32 bits)
    )

```

Vowelizer models:

|Model|Model ID|Paper|Repo|Architecture|
|-----|--------|---------|----|--|
|CATT|catt_eo|[arxiv](https://arxiv.org/abs/2407.03236)|[github](https://github.com/abjadai/catt)|Transformer Encoder|
|Shakkelha|shakkelha|[arxiv](https://arxiv.org/abs/1911.03531)|[github](https://github.com/AliOsm/shakkelha)|Bi-LSTM|
|Shakkala|shakkala|-|[github](https://github.com/Barqawiz/Shakkala)|Bi-LSTM|


References:

The vocoder `vocos44` was converted from ([patriotyk/vocos-mel-hifigan-compat-44100khz](https://huggingface.co/patriotyk/vocos-mel-hifigan-compat-44100khz)).

The vowelizer `catt_eo` was converted from https://github.com/abjadai/catt/releases/tag/v2 *best_eo_mlm_ns_epoch_193.pt* (License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))


![DALL·E 2025-03-14 18 56 01 - A surreal digital painting of a camel in a vast desert, with a futuristic speaker embedded in its mouth, symbolizing text-to-speech technology  The ca](https://github.com/user-attachments/assets/bcd31436-1e76-4432-9072-d0695f4d87e0)
