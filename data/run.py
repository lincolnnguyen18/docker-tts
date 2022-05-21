lang = 'Japanese'
type = "tsukuyomi_full_band_vits_prosody"
tag = f"kan-bayashi/{type}"
#@param ["kan-bayashi/jsut_tacotron2", "kan-bayashi/jsut_transformer", "kan-bayashi/jsut_fastspeech", "kan-bayashi/jsut_fastspeech2", "kan-bayashi/jsut_conformer_fastspeech2", "kan-bayashi/jsut_conformer_fastspeech2_accent", "kan-bayashi/jsut_conformer_fastspeech2_accent_with_pause", "kan-bayashi/jsut_vits_accent_with_pause", "kan-bayashi/jsut_full_band_vits_accent_with_pause", "kan-bayashi/jsut_tacotron2_prosody", "kan-bayashi/jsut_transformer_prosody", "kan-bayashi/jsut_conformer_fastspeech2_tacotron2_prosody", "kan-bayashi/jsut_vits_prosody", "kan-bayashi/jsut_full_band_vits_prosody", "kan-bayashi/jvs_jvs010_vits_prosody", "kan-bayashi/tsukuyomi_full_band_vits_prosody"] {type:"string"}
vocoder_tag = 'none' #@param ["none", "parallel_wavegan/jsut_parallel_wavegan.v1", "parallel_wavegan/jsut_multi_band_melgan.v2", "parallel_wavegan/jsut_style_melgan.v1", "parallel_wavegan/jsut_hifigan.v1"] {type:"string"}

from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import time
import torch
# print(torch.cuda.is_available())

text2speech = Text2Speech.from_pretrained(
    model_tag=str_or_none(tag),
    vocoder_tag=str_or_none(vocoder_tag),
    device="cuda",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

# decide the input sentence by yourself
# print(f"Input your favorite sentence in {lang}.")
# x = input()
x = 'やけにテンションが高い。さっきまで死にかけていたとは思えない。'

# synthesis
with torch.no_grad():
    start = time.time()
    wav = text2speech(x)["wav"]
rtf = (time.time() - start) / (len(wav) / text2speech.fs)
print(f"RTF = {rtf:5f}")

# let us listen to generated samples
from IPython.display import display, Audio
# display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))

# export wav to file
with open (f"{type}.wav", "wb") as f:
    f.write(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs).data)