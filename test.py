# %%
from tts_arabic import tts

# %%
text = "السَّلامُ عَلَيكُم يَا صَدِيقِي."
# text = "أَلِف با تا ثا جِيم حَا خَا دَال ذَال را زَاي سِين شِين صَاد ضَاد طَا ظَا عَين غَين فَا قَاف كَاف لَام مِيم نُون هَا وَاو يَا."
wave = tts(text, speaker=1, play=True)

# %% Buckwalter transliteration
text = ">ls~alAmu Ealaykum yA Sadiyqiy."
wave = tts(text, speaker=0, play=True)

# %% Unvocalized input
text_unvoc = "القهوة مشروب يعد من بذور البن المحمصة"
wave = tts(text_unvoc, play=True, vowelizer='catt_eo')

# %% Inference options

text = "السَّلامُ عَلَيكُم يَا صَدِيقِي."
wave = tts(
    text, # input text
    speaker = 1, # speaker id; choose between 0,1,2,3
    pace = 1, # speaker pace
    denoise = 0.005, # vocoder denoising strength
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

# %%
