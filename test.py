# %%
from tts_arabic import tts

# %%
text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
wave = tts(text, pace=0.9, speaker=2, play=True)

# %% Buckwalter transliteration
text = ">als~alAmu Ealaykum yA Sadiyqiy."
wave = tts(text, speaker=0, play=True)

# %% Unvocalized input
text_unvoc = "القهوة مشروب يعد من بذور البن المحمصة"
wave = tts(text_unvoc, play=True, vowelizer='shakkelha')

# %%