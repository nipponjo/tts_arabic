# Licensing scope and model provenance

## Original project code

The MIT license in `LICENSE` applies to original code and documentation authored
by nipponjo in this repository, except where a file or component specifies
different terms. Existing third-party copyright and license notices remain in
effect. The MIT grant does not relicense third-party code, pretrained weights,
ONNX model files, training datasets, or generated training audio.

See [ThirdPartyLicenses](tts_arabic/ThirdPartyLicenses) and the license files
alongside third-party components for their terms. The presence of a model
download URL does not grant an MIT license to that model.

## Multi-speaker model training sources

The existing multi-speaker releases (`fastpitch`, `mixer128`, and `mixer80`)
were trained using four voices:

| Training voice | Source and terms |
| --- | --- |
| ASC voice | Arabic Speech Corpus by Nawar Halabi, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). |
| Three additional voices | Synthetic training speech generated using Coqui XTTS-v2, made available under the Coqui Public Model License 1.0.0 (CPML), a custom noncommercial license. |

Arabic Speech Corpus source: https://en.arabicspeechcorpus.com/

ASC license: https://creativecommons.org/licenses/by/4.0/

The ASC recordings/transcripts were used for neural TTS training and model
conversion. No endorsement by the corpus authors or upstream developers is implied.

XTTS-v2 source: https://huggingface.co/coqui/XTTS-v2

CPML text: https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt

The CPML covers XTTS-v2 and its outputs and restricts their use to noncommercial
purposes. It expressly excludes training other models for commercial use from
its definition of noncommercial purposes. The complete CPML terms are included
in `tts_arabic/ThirdPartyLicenses`; they must be considered alongside any
model-specific release terms. This notice does not sublicense XTTS-v2 or remove
any CPML restriction.

Accordingly, these mixed-training multi-speaker releases are provided for
noncommercial use only unless separate permissions covering the XTTS-derived
training material and intended use have been obtained. Selecting the ASC voice
in a mixed-training checkpoint does not turn that checkpoint into an ASC-only
model or establish permission for commercial use.

## Third-party frontend and other components

`tts_arabic/text/phonetise_buckwalter.py` is adapted from Nawar Halabi's
Arabic-Phonetiser and retains its CC BY-NC 4.0 license. The project's MIT license
does not remove that component's noncommercial condition. The combined package
must therefore not be described as entirely MIT-licensed or as unrestricted for
commercial use.

Frontend source: https://github.com/nawarhalabi/Arabic-Phonetiser

Frontend license: https://creativecommons.org/licenses/by-nc/4.0/

Neural vocoders, diacritizers, and model implementations have their own notices
and terms, as recorded in `tts_arabic/ThirdPartyLicenses` and component license
files. When distributing model files separately from this package, include the
applicable licenses, attribution, provenance, and restrictions with that release.
