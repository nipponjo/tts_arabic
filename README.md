[[Pytorch repo]](https://github.com/nipponjo/tts-arabic-pytorch) 
[[Samples]](https://nipponjo.github.io/tts-arabic-speakers) 
[[huggingface](https://huggingface.co/nipponjo/tts-arabic-onnx) |
[online demo](https://huggingface.co/spaces/nipponjo/tts-arabic-demo)]
[[Flutter app]](https://github.com/nipponjo/tts-arabic-flutter)
[[arXiv]](https://arxiv.org/abs/2512.00937)

Arabic TTS models (FastPitch, MixerTTS) from the [tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch) repo in the ONNX format — usable as a Python package for offline speech synthesis.

Audio samples can be found [here](https://nipponjo.github.io/tts-arabic-speakers).

**Licensing**

Original project code is licensed under [MIT](LICENSE), subject to the
third-party exceptions in [NOTICE.md](NOTICE.md) and
[ThirdPartyLicenses](tts_arabic/ThirdPartyLicenses). Model weights and datasets
are licensed separately; the MIT code license does not cover them.

The multi-speaker models use one Arabic Speech Corpus voice (CC BY 4.0) and
three voices generated with XTTS-v2 (Coqui Public Model License 1.0.0).
These mixed-training models are for noncommercial use unless separately
authorized. The Arabic phonetiser also retains its CC BY-NC 4.0 terms.

**Install with**
```
pip install git+https://github.com/nipponjo/tts_arabic.git
```


**Examples**
```python

# %%
from tts_arabic import tts

# %%
text = "السَّلامُ عَلَيكُم يَا صَدِيقِي."
wave = tts(text, speaker=2, pace=0.9, play=True)

# %% Buckwalter transliteration
text = ">ls~alAmu Ealaykum yA Sadiyqiy."
wave = tts(text, speaker=0, play=True)

# %% Unvocalized input
text_unvoc = "القهوة مشروب يعد من بذور البن المحمصة"
wave = tts(text_unvoc, play=True, vowelizer='shakkelha')

```

**Pretrained models**

|Model|Model ID|Type|#params|Paper|Output|
|-------|---|---|------|----|----|
|FastPitch|fastpitch|Text->Mel|46.3M|[arxiv](https://arxiv.org/abs/2006.06873)|Mel (80 bins)|
|MixerTTS|mixer128|Text->Mel|2.9M|[arxiv](https://arxiv.org/abs/2110.03584)|Mel (80 bins)|
|MixerTTS|mixer80|Text->Mel|1.5M|[arxiv](https://arxiv.org/abs/2110.03584)|Mel (80 bins)|
|MixerTTS (ASC, single speaker)|mixer128-asc|Text->Mel|3.15M|—|Mel (80 bins)|
|HiFi-GAN|hifigan|Vocoder|13.9M|[arxiv](https://arxiv.org/abs/2010.05646)|Wave (22.05kHz)|
|Vocos|vocos|Vocoder|13.4M|[arxiv](https://arxiv.org/abs/2306.00814)|Wave (22.05kHz)|
|Vocos|vocos44|Vocoder|14.0M|[arxiv](https://arxiv.org/abs/2306.00814)|Wave (44.1kHz)|

The sequence of transformations is as follows:

*Text* &rarr; Phonemizer &rarr; *Phonemes* &rarr; Tokenizer &rarr; *Token Ids* &rarr; **Text->Mel** model &rarr; *Mel spectrogram* &rarr; **Vocoder** model &rarr; *Wave*

The `Text->Mel` models map token ids to mel frames. All models use the 80 bin configuration proposed by [HiFi-GAN](https://github.com/jik876/hifi-gan). This mel spectrogram contains frequencies up to 8kHz. The `vocoder` models map the mel spectrogram to a waveform. The vocoders with `vocoder_id` `hifigan` and `vocos` artificially extend the bandwidth to 11025Hz, and `vocos44` to 22050Hz. Samples for comparing the models can be found [here](https://nipponjo.github.io/tts-arabic-speakers/#models-cmp).

**Local MixerTTS exports and model metadata**

`mixer128-asc` loads `tts_arabic/data/mixer128-asc.onnx`. This is a local export;
there is no download URL for it yet. From the sibling `mixer-tts-pytorch` repo:

```bash
python to_onnx_mixer.py --config configs/nawar_fp_adv_norm.yaml --output ../tts_arabic/tts_arabic/data/mixer128-asc.onnx
```

Use it with:

```python
from tts_arabic import tts, get_model

wave = tts(text, model_id='mixer128-asc', save_to='speech.wav')
model = get_model('mixer128-asc', cuda=None)
print(model.sample_rate, model.metadata)
```

The default vocoder and output sample rate come from the ONNX metadata
(currently `vocos44`, 44100 Hz). Playback and WAV headers use the resolved
pipeline sample rate. Passing `vocoder_id='vocos'` explicitly continues to
produce 22050 Hz audio. `tts.sr` contains the rate of the most recently used
pipeline.

The versioned JSON object in ONNX `custom_metadata_map['tts_arabic']` stores
`schema_version`, `model_id`, `architecture`, `language`, ordered `symbols`,
`vocoder_id`, `sample_rate`, `mel`, speaker/emotion counts, dimension, F0
statistics, and checkpoint provenance. The runtime reads the symbol-to-ID
mapping and default vocoder settings from the model. `mel.sample_rate` is the
analysis rate, while `sample_rate` is the preferred waveform rate. If a vocoder
also contains this metadata, its `sample_rate` takes precedence.

Existing models without metadata retain their original token IDs, default
HiFi-GAN vocoder, and legacy vocoder sample rates. The runtime feeds only the
inputs present in each ONNX graph, supporting both legacy multi-speaker models
and new single-speaker exports whose speaker/emotion inputs were removed.

**Manuscript**

More information about how the models were trained can be found in the manuscript *Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis* ([arXiv](https://arxiv.org/abs/2512.00937) | [ResearchGate](https://www.researchgate.net/publication/398225096_Arabic_TTS_with_FastPitch_Reproducible_Baselines_Adversarial_Training_and_Oversmoothing_Analysis)).


**TTS options**
```python
from tts_arabic import tts

text = "السَّلامُ عَلَيكُم يَا صَدِيقِي."
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
    bits_per_sample = 32, # when save_to is specified (8, 16, 24 or 32 bits)
    )

```

**Vowelizer models**

|Model|Model ID|Paper|Repo|Architecture|
|-----|--------|---------|----|--|
|CATT|catt_eo|[arxiv](https://arxiv.org/abs/2407.03236)|[github](https://github.com/abjadai/catt)|Transformer Encoder|
|Shakkelha|shakkelha|[arxiv](https://arxiv.org/abs/1911.03531)|[github](https://github.com/AliOsm/shakkelha)|Bi-LSTM|
|Shakkala|shakkala|-|[github](https://github.com/Barqawiz/Shakkala)|Bi-LSTM|


**References**

The vocoder `vocos44` was converted from ([patriotyk/vocos-mel-hifigan-compat-44100khz](https://huggingface.co/patriotyk/vocos-mel-hifigan-compat-44100khz)).

The vowelizer `catt_eo` was converted from https://github.com/abjadai/catt/releases/tag/v2 *best_eo_mlm_ns_epoch_193.pt* (License: [Apache-2.0](https://github.com/abjadai/catt?tab=Apache-2.0-1-ov-file#readme))


![DALL·E 2025-03-14 18 56 01 - A surreal digital painting of a camel in a vast desert, with a futuristic speaker embedded in its mouth, symbolizing text-to-speech technology  The ca](https://github.com/user-attachments/assets/bcd31436-1e76-4432-9072-d0695f4d87e0)
